"""Trusted Authority, pseudonym lifecycle and vehicle credential wallets.

BC-PAFL authentication (Sec. III-A, Fig. 2 "vehicle authentication"):

1. **Registration.**  A vehicle's real identity is enrolled (MVD role).  The TA
   never publishes it.
2. **Pseudonym issuance.**  The TA, in coordination with the KGC, generates
   fresh pseudonyms ``AID = (k*P, ID xor H0(t*k*P || T_pub))`` with KGC partial
   keys.  The vehicle completes each key pair with its own secret value (the TA
   never learns it) and the TA records every pseudonym's public key, validity
   window and seeded reputation on the blockchain.
3. **Authentication.**  Each round the vehicle presents a pseudonym chosen at
   random from its valid pool; its beacon is signed with that pseudonym's key
   and embeds the RSU's fresh round nonce (proof of possession + freshness).  The RSU checks the ledger for
   authenticity, freshness and revocation status -- the ledger holds no real
   identity.
4. **Conditional traceability.**  Only the TA (holding ``t``) can recover the
   real identity from an AID, which it does when RSUs report misbehaviour, and
   it revokes every pseudonym of an identity that keeps misbehaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .blockchain import (
    BlockchainNetwork, Ledger, TX_PSEUDONYM_BATCH, TX_PSEUDONYM_REVOKE,
)
from .crypto.certificateless import (
    P, Aid, KeyGenerationCenter, KeyPair, PublicKey, SecurityError, h0_mask,
    random_scalar, sha256_digest,
)


def _id_token(real_id: str) -> bytes:
    raw = real_id.encode("utf-8")
    return sha256_digest(raw) if len(raw) > 32 else raw.ljust(32, b"\x00")


class TrustedAuthority:
    """TA (+ MVD registry) working with a KGC and the permissioned ledger."""

    ta_id = "TA"

    def __init__(self, kgc: KeyGenerationCenter) -> None:
        self.kgc = kgc
        self._t = random_scalar()
        self.T_pub = self._t * P
        self._mvd: Dict[bytes, str] = {}
        self._issued: Dict[str, List[str]] = {}        # real id -> pseudonym ids
        self._revoked_ids: Set[str] = set()
        # Beta reputation per real identity, carried across pseudonyms.
        self._reputation: Dict[str, List[float]] = {}
        self._feedback_seen: Dict[str, Tuple[float, float, int]] = {}
        self._anomalies: Dict[str, int] = {}
        self.chain: Optional[BlockchainNetwork] = None
        self.keypair: Optional[KeyPair] = None
        self.trace_log: List[Tuple[int, str, str]] = []  # (round, pseudonym, real id)

    # ------------------------------------------------------------------
    # Registration and key material
    # ------------------------------------------------------------------
    def enroll(self, real_id: str) -> None:
        if not real_id:
            raise SecurityError("empty identity")
        token = _id_token(real_id)
        if self._mvd.get(token, real_id) != real_id:
            raise SecurityError("identity token collision")
        self._mvd[token] = real_id
        self._reputation.setdefault(real_id, [0.0, 0.0])

    def _generate_aid(self, real_id: str) -> Aid:
        token = _id_token(real_id)
        if self._mvd.get(token) != real_id:
            raise SecurityError(f"{real_id!r} is not enrolled")
        aid_point = random_scalar() * P
        mask = h0_mask(self._t * aid_point, self.T_pub)
        return aid_point, bytes(a ^ b for a, b in zip(token, mask))

    def trace(self, public_key: PublicKey) -> str:
        """Conditional traceability: recover the real identity behind an AID."""
        aid_point, aid_token = public_key.aid
        mask = h0_mask(self._t * aid_point, self.T_pub)
        real_id = self._mvd.get(bytes(a ^ b for a, b in zip(aid_token, mask)))
        if real_id is None:
            raise SecurityError("AID does not map to an enrolled identity")
        return real_id

    def issue_infrastructure_key(self, entity_id: str) -> KeyPair:
        """Long-term key for the TA itself, the base station or an RSU."""
        self.enroll(entity_id)
        aid = self._generate_aid(entity_id)
        return KeyPair(aid, self.kgc.extract_partial_private_key(aid), self.kgc.P_pub)

    def attach_chain(self, chain: BlockchainNetwork, keypair: KeyPair) -> None:
        self.chain = chain
        self.keypair = keypair

    # ------------------------------------------------------------------
    # Pseudonyms
    # ------------------------------------------------------------------
    def reputation(self, real_id: str) -> Tuple[float, float]:
        a, b = self._reputation.get(real_id, [0.0, 0.0])
        return a, b

    def is_revoked(self, real_id: str) -> bool:
        return real_id in self._revoked_ids

    def provision(self, wallet: "VehicleWallet", round_num: int, count: int,
                  lifetime: int) -> List[str]:
        """Issue ``count`` fresh pseudonyms valid for rounds [round, round+lifetime-1]."""
        if self.chain is None or self.keypair is None:
            raise RuntimeError("TA is not attached to the blockchain")
        real_id = wallet.real_id
        if real_id in self._revoked_ids:
            return []
        records, new_ids = [], []
        alpha, beta = self.reputation(real_id)
        for _ in range(count):
            aid = self._generate_aid(real_id)
            partial = self.kgc.extract_partial_private_key(aid)
            public_key = wallet.complete_key(aid, partial, round_num, round_num + lifetime - 1)
            if self.trace(public_key) != real_id or not public_key.reconstruct_ok():
                raise SecurityError("issued pseudonym failed the TA cross-check")
            pid = public_key.fingerprint()
            records.append({
                "pseudonym_id": pid,
                "public_key": public_key.to_bytes().hex(),
                "valid_from": round_num,
                "valid_until": round_num + lifetime - 1,
                "trust_alpha": round(alpha, 4),
                "trust_beta": round(beta, 4),
            })
            new_ids.append(pid)
        self._issued.setdefault(real_id, []).extend(new_ids)
        self.chain.submit(self.ta_id, TX_PSEUDONYM_BATCH, {"records": records})
        return new_ids

    def revoke(self, real_id: str, round_num: int, reason: str) -> List[str]:
        if self.chain is None:
            raise RuntimeError("TA is not attached to the blockchain")
        self._revoked_ids.add(real_id)
        ledger = self.chain.reference()
        pids = [pid for pid in self._issued.get(real_id, [])
                if pid in ledger.pseudonyms and not ledger.pseudonyms[pid]["revoked"]]
        if pids:
            self.chain.submit(self.ta_id, TX_PSEUDONYM_REVOKE,
                              {"pseudonym_ids": pids, "reason": reason, "round": round_num})
        return pids

    def process_feedback(self, ledger: Ledger, round_num: int, forgetting: float,
                         revoke_after: int) -> List[str]:
        """Fold new on-chain RSU feedback into per-identity reputation.

        Tracing uses the AID recovery trapdoor, not a lookup table, so it works
        for any pseudonym an RSU reports.  Returns real ids revoked this round.
        """
        for rep in self._reputation.values():
            rep[0] *= forgetting
            rep[1] *= forgetting
        revoked = []
        for pid, rec in ledger.pseudonyms.items():
            seen = self._feedback_seen.get(pid, (0.0, 0.0, 0))
            d_pos = rec["positive"] - seen[0]
            d_neg = rec["negative"] - seen[1]
            d_anom = rec["anomalies"] - seen[2]
            if d_pos <= 0 and d_neg <= 0 and d_anom <= 0:
                continue
            self._feedback_seen[pid] = (rec["positive"], rec["negative"], rec["anomalies"])
            real_id = self.trace(rec["public_key_obj"])
            rep = self._reputation.setdefault(real_id, [0.0, 0.0])
            rep[0] += max(d_pos, 0.0)
            rep[1] += max(d_neg, 0.0)
            if d_anom > 0:
                self.trace_log.append((round_num, pid, real_id))
                self._anomalies[real_id] = self._anomalies.get(real_id, 0) + d_anom
                if (self._anomalies[real_id] >= revoke_after
                        and real_id not in self._revoked_ids):
                    self.revoke(real_id, round_num, "repeated anomalous model updates")
                    revoked.append(real_id)
        return revoked


@dataclass
class Pseudonym:
    pseudonym_id: str
    keypair: KeyPair
    valid_from: int
    valid_until: int


class VehicleWallet:
    """Credential store in the vehicle's tamper-proof device (TPD)."""

    def __init__(self, real_id: str, kgc_p_pub) -> None:
        self.real_id = real_id
        self.pseudonyms: List[Pseudonym] = []
        self._kgc_p_pub = kgc_p_pub

    def complete_key(self, aid: Aid, partial_key, valid_from: int, valid_until: int) -> PublicKey:
        """Finish a key pair with a locally generated secret value x (KeyPair
        validates the KGC partial key before accepting it)."""
        keypair = KeyPair(aid, partial_key, self._kgc_p_pub)
        self.pseudonyms.append(Pseudonym(keypair.public_key.fingerprint(), keypair,
                                         valid_from, valid_until))
        return keypair.public_key

    def valid(self, round_num: int) -> List[Pseudonym]:
        return [p for p in self.pseudonyms if p.valid_from <= round_num <= p.valid_until]

    def prune(self, round_num: int) -> None:
        self.pseudonyms = [p for p in self.pseudonyms if p.valid_until >= round_num]

    def select(self, round_num: int, rng: np.random.Generator) -> Optional[Pseudonym]:
        """Dynamic pseudonym: pick one valid pseudonym uniformly at random."""
        pool = self.valid(round_num)
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]
