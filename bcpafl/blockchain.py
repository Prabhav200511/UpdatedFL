"""Permissioned blockchain for pseudonym management (BC-PAFL, Fig. 1).

The ledger records pseudonym validity and revocation status, trust feedback
and global-model commitments, so RSUs/OBUs can independently check the
authenticity, freshness and current status of a pseudonym without learning the
real identity behind it.

Design
------
* **Validators** (Proof-of-Authority): the TA, the base station and every RSU.
  Each has a long-term certificateless key recorded in the genesis block.
* **Transactions** are signed by their submitter and authorised by type:
  pseudonym registration/revocation only by the TA, trust feedback only by
  RSUs, model commitments only by the base station.
* **Blocks** carry a Merkle root over transaction hashes and link to the
  previous header hash.  A block is committed only with endorsements from a
  quorum of ``floor(2n/3) + 1`` validators, each of which re-validated it.
* **Replicas**: every validator keeps its own :class:`Ledger` and appends a
  block only after independently validating header, links, Merkle root,
  endorsements and every transaction.  ``Ledger.verify_chain`` re-checks the
  whole chain, so tampering with any stored block is detected.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .crypto.certificateless import (
    KeyPair, PublicKey, SecurityError, Verifier, sha256_digest,
    signature_from_bytes, signature_to_bytes,
)

TX_INFRA_REGISTER = "INFRA_REGISTER"
TX_PSEUDONYM_BATCH = "PSEUDONYM_BATCH"
TX_PSEUDONYM_REVOKE = "PSEUDONYM_REVOKE"
TX_TRUST_FEEDBACK = "TRUST_FEEDBACK"
TX_MODEL_COMMIT = "MODEL_COMMIT"

ROLE_TA = "TA"
ROLE_BS = "BS"
ROLE_RSU = "RSU"

_AUTHORISED_ROLES = {
    TX_INFRA_REGISTER: {ROLE_TA},
    TX_PSEUDONYM_BATCH: {ROLE_TA},
    TX_PSEUDONYM_REVOKE: {ROLE_TA},
    TX_TRUST_FEEDBACK: {ROLE_RSU},
    TX_MODEL_COMMIT: {ROLE_BS},
}


class LedgerError(ValueError):
    pass


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def merkle_root(hashes: Sequence[str]) -> str:
    if not hashes:
        return sha256_digest(b"BCPAFL/merkle/empty").hex()
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256_digest(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0].hex()


@dataclass
class Transaction:
    tx_type: str
    submitter: str
    payload: Dict[str, Any]
    nonce: int
    signature: str = ""

    def body(self) -> Dict[str, Any]:
        return {"type": self.tx_type, "submitter": self.submitter,
                "payload": self.payload, "nonce": self.nonce}

    def tx_hash(self) -> str:
        return sha256_digest(b"BCPAFL/tx/v1" + canonical(self.body())).hex()

    def sign(self, keypair: KeyPair) -> "Transaction":
        self.signature = signature_to_bytes(keypair.sign(bytes.fromhex(self.tx_hash()))).hex()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {**self.body(), "signature": self.signature}


@dataclass
class Block:
    index: int
    prev_hash: str
    round: int
    proposer: str
    merkle: str
    transactions: List[Transaction]
    proposer_signature: str = ""
    endorsements: Dict[str, str] = field(default_factory=dict)

    def header(self) -> Dict[str, Any]:
        return {"index": self.index, "prev": self.prev_hash, "round": self.round,
                "proposer": self.proposer, "merkle": self.merkle}

    def block_hash(self) -> str:
        return sha256_digest(b"BCPAFL/block/v1" + canonical(self.header())).hex()

    def size_bytes(self) -> int:
        return len(canonical({"header": self.header(),
                              "txs": [t.to_dict() for t in self.transactions],
                              "sig": self.proposer_signature,
                              "endorsements": self.endorsements}))


class Ledger:
    """One validator's replica of the chain plus its derived world state."""

    def __init__(self, owner: str, p_pub, genesis_validators: Mapping[str, Tuple[str, bytes]]):
        self.owner = owner
        self.verifier = Verifier(p_pub)
        # validator id -> (role, PublicKey)
        self.validators: Dict[str, Tuple[str, PublicKey]] = {
            vid: (role, PublicKey.from_bytes(pk)) for vid, (role, pk) in genesis_validators.items()}
        self._genesis_validators = dict(genesis_validators)
        self.blocks: List[Block] = []
        self._validated_proposal: Optional[Tuple[int, str]] = None
        self._reset_state()

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.infra: Dict[str, Dict[str, Any]] = {}
        self.pseudonyms: Dict[str, Dict[str, Any]] = {}
        self.model_commits: Dict[int, Dict[str, Any]] = {}
        self.seen_nonces: set = set()

    def quorum(self) -> int:
        return (2 * len(self.validators)) // 3 + 1

    @property
    def height(self) -> int:
        return len(self.blocks)

    def head_hash(self) -> str:
        return self.blocks[-1].block_hash() if self.blocks else "0" * 64

    def pseudonym_status(self, pseudonym_id: str, round_num: int) -> Tuple[bool, str]:
        """Authenticity, freshness and revocation check for one pseudonym."""
        record = self.pseudonyms.get(pseudonym_id)
        if record is None:
            return False, "unregistered"
        if record["revoked"]:
            return False, "revoked"
        if round_num < record["valid_from"]:
            return False, "not_yet_valid"
        if round_num > record["valid_until"]:
            return False, "expired"
        return True, "valid"

    def pseudonym_public_key(self, pseudonym_id: str) -> Optional[PublicKey]:
        record = self.pseudonyms.get(pseudonym_id)
        return None if record is None else record["public_key_obj"]

    def infra_public_key(self, entity_id: str) -> Optional[PublicKey]:
        if entity_id in self.validators:
            return self.validators[entity_id][1]
        record = self.infra.get(entity_id)
        return None if record is None else PublicKey.from_bytes(bytes.fromhex(record["public_key"]))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_tx_semantics(self, tx: Transaction, pending: Dict[str, Dict[str, Any]]) -> None:
        role_pk = self.validators.get(tx.submitter)
        if role_pk is None:
            raise LedgerError(f"unknown submitter {tx.submitter}")
        role = role_pk[0]
        if role not in _AUTHORISED_ROLES.get(tx.tx_type, set()):
            raise LedgerError(f"{role} may not submit {tx.tx_type}")
        if (tx.submitter, tx.nonce) in self.seen_nonces:
            raise LedgerError("replayed transaction nonce")
        p = tx.payload
        if tx.tx_type == TX_PSEUDONYM_BATCH:
            for rec in p["records"]:
                pid = rec["pseudonym_id"]
                pk = PublicKey.from_bytes(bytes.fromhex(rec["public_key"]))
                if pk.fingerprint() != pid:
                    raise LedgerError("pseudonym id does not match its public key")
                if not pk.reconstruct_ok():
                    raise LedgerError("pseudonym public key fails reconstruction")
                if pid in self.pseudonyms or pid in pending:
                    raise LedgerError("duplicate pseudonym registration")
                if rec["valid_until"] < rec["valid_from"]:
                    raise LedgerError("invalid validity window")
                pending[pid] = rec
        elif tx.tx_type == TX_PSEUDONYM_REVOKE:
            for pid in p["pseudonym_ids"]:
                if pid not in self.pseudonyms and pid not in pending:
                    raise LedgerError("cannot revoke an unknown pseudonym")
        elif tx.tx_type == TX_TRUST_FEEDBACK:
            for entry in p["entries"]:
                if entry["pseudonym_id"] not in self.pseudonyms:
                    raise LedgerError("feedback for an unknown pseudonym")
        elif tx.tx_type == TX_INFRA_REGISTER:
            PublicKey.from_bytes(bytes.fromhex(p["public_key"]))

    def validate_block(self, block: Block, require_endorsements: bool = True) -> None:
        if block.index != self.height:
            raise LedgerError("block index does not extend this replica")
        if block.prev_hash != self.head_hash():
            raise LedgerError("block does not link to the current head")
        if block.merkle != merkle_root([t.tx_hash() for t in block.transactions]):
            raise LedgerError("Merkle root mismatch")
        proposer = self.validators.get(block.proposer)
        if proposer is None:
            raise LedgerError("proposer is not an authorised validator")
        header_digest = bytes.fromhex(block.block_hash())
        checks = [(header_digest, signature_from_bytes(bytes.fromhex(block.proposer_signature)),
                   proposer[1])]
        for tx in block.transactions:
            checks.append((bytes.fromhex(tx.tx_hash()),
                           signature_from_bytes(bytes.fromhex(tx.signature)),
                           self.validators[tx.submitter][1] if tx.submitter in self.validators
                           else _raise(LedgerError(f"unknown submitter {tx.submitter}"))))
        if require_endorsements:
            valid_endorsers = [v for v in block.endorsements if v in self.validators]
            if len(set(valid_endorsers)) < self.quorum():
                raise LedgerError("block lacks a validator quorum")
            for vid in valid_endorsers:
                checks.append((header_digest,
                               signature_from_bytes(bytes.fromhex(block.endorsements[vid])),
                               self.validators[vid][1]))
        if not all(self.verifier.verify_many(checks)):
            raise LedgerError("invalid signature in block")
        pending: Dict[str, Dict[str, Any]] = {}
        nonces = set()
        for tx in block.transactions:
            if (tx.submitter, tx.nonce) in nonces:
                raise LedgerError("duplicate transaction in block")
            nonces.add((tx.submitter, tx.nonce))
            self._check_tx_semantics(tx, pending)

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------
    def _apply(self, block: Block) -> None:
        for tx in block.transactions:
            self.seen_nonces.add((tx.submitter, tx.nonce))
            p = tx.payload
            if tx.tx_type == TX_INFRA_REGISTER:
                self.infra[p["entity_id"]] = dict(p)
            elif tx.tx_type == TX_PSEUDONYM_BATCH:
                for rec in p["records"]:
                    self.pseudonyms[rec["pseudonym_id"]] = {
                        **rec, "revoked": False, "revoked_round": None,
                        "positive": 0.0, "negative": 0.0, "anomalies": 0,
                        "public_key_obj": PublicKey.from_bytes(bytes.fromhex(rec["public_key"])),
                    }
            elif tx.tx_type == TX_PSEUDONYM_REVOKE:
                for pid in p["pseudonym_ids"]:
                    self.pseudonyms[pid]["revoked"] = True
                    self.pseudonyms[pid]["revoked_round"] = block.round
            elif tx.tx_type == TX_TRUST_FEEDBACK:
                for entry in p["entries"]:
                    rec = self.pseudonyms[entry["pseudonym_id"]]
                    rec["positive"] += float(entry["positive"])
                    rec["negative"] += float(entry["negative"])
                    rec["anomalies"] += int(entry.get("anomalies", 0))
            elif tx.tx_type == TX_MODEL_COMMIT:
                self.model_commits[int(p["round"])] = dict(p)

    def endorse_check(self, block: Block) -> None:
        """Full validation of a proposal; remembered so the commit only needs the quorum."""
        self.validate_block(block, require_endorsements=False)
        self._validated_proposal = (block.index, block.block_hash())

    def append(self, block: Block) -> None:
        if self._validated_proposal == (block.index, block.block_hash()) \
                and block.index == self.height and block.prev_hash == self.head_hash():
            # Header hash commits to the Merkle root, so the content is the one
            # this replica already validated; only the endorsement quorum is new.
            header_digest = bytes.fromhex(block.block_hash())
            endorsers = [v for v in block.endorsements if v in self.validators]
            if len(set(endorsers)) < self.quorum():
                raise LedgerError("block lacks a validator quorum")
            checks = [(header_digest, signature_from_bytes(bytes.fromhex(block.endorsements[v])),
                       self.validators[v][1]) for v in endorsers]
            if not all(self.verifier.verify_many(checks)):
                raise LedgerError("invalid endorsement signature")
            if block.merkle != merkle_root([t.tx_hash() for t in block.transactions]):
                raise LedgerError("Merkle root mismatch")
        else:
            self.validate_block(block)
        self._validated_proposal = None
        self._apply(block)
        self.blocks.append(block)

    def verify_chain(self) -> bool:
        """Re-validate every stored block from genesis (tamper detection)."""
        replay = Ledger(self.owner, self.verifier.P_pub, self._genesis_validators)
        try:
            for block in self.blocks:
                replay.append(block)
        except (LedgerError, SecurityError, KeyError, ValueError, TypeError):
            return False
        return replay.head_hash() == self.head_hash()


def _raise(exc: Exception):
    raise exc


class BlockchainNetwork:
    """Validator set, shared mempool, round-robin PoA proposing and replication."""

    def __init__(self, p_pub, validators: Mapping[str, Tuple[str, KeyPair]]):
        if not validators:
            raise ValueError("a blockchain needs at least one validator")
        self.p_pub = p_pub
        self._keys = {vid: kp for vid, (_, kp) in validators.items()}
        genesis = {vid: (role, kp.public_key.to_bytes()) for vid, (role, kp) in validators.items()}
        self.replicas: Dict[str, Ledger] = {vid: Ledger(vid, p_pub, genesis) for vid in validators}
        self.order = sorted(validators)
        self.mempool: List[Transaction] = []
        self._nonces: Dict[str, int] = {vid: 0 for vid in validators}
        self.stats = {"blocks": 0, "transactions": 0, "rejected_blocks": 0, "bytes": 0}

    def submit(self, submitter: str, tx_type: str, payload: Dict[str, Any]) -> Transaction:
        if submitter not in self._keys:
            raise LedgerError(f"{submitter} is not a validator")
        self._nonces[submitter] += 1
        tx = Transaction(tx_type, submitter, copy.deepcopy(payload), self._nonces[submitter])
        tx.sign(self._keys[submitter])
        self.mempool.append(tx)
        return tx

    def reference(self) -> Ledger:
        return self.replicas[self.order[0]]

    def ledger(self, owner: str) -> Ledger:
        return self.replicas[owner]

    def commit_pending(self, round_num: int) -> Optional[Block]:
        """Propose, endorse and replicate one block with every pending tx."""
        if not self.mempool:
            return None
        reference = self.reference()
        proposer = self.order[reference.height % len(self.order)]
        txs, self.mempool = self.mempool, []
        block = Block(reference.height, reference.head_hash(), round_num, proposer,
                      merkle_root([t.tx_hash() for t in txs]), txs)
        digest = bytes.fromhex(block.block_hash())
        block.proposer_signature = signature_to_bytes(self._keys[proposer].sign(digest)).hex()
        # Each validator validates the proposal on its own replica before endorsing.
        for vid in self.order:
            try:
                self.replicas[vid].endorse_check(block)
            except (LedgerError, SecurityError, KeyError, ValueError) as exc:
                if vid == proposer:
                    self.stats["rejected_blocks"] += 1
                    raise LedgerError(f"proposer {proposer} built an invalid block: {exc}") from exc
                continue
            block.endorsements[vid] = signature_to_bytes(self._keys[vid].sign(digest)).hex()
        if len(block.endorsements) < reference.quorum():
            self.stats["rejected_blocks"] += 1
            raise LedgerError("proposal failed to reach a validator quorum")
        for vid in self.order:
            # Independent copies: a replica never shares mutable state with another.
            self.replicas[vid].append(copy.deepcopy(block))
        self.stats["blocks"] += 1
        self.stats["transactions"] += len(txs)
        self.stats["bytes"] += block.size_bytes()
        return block

    def replicas_consistent(self) -> bool:
        heads = {ledger.head_hash() for ledger in self.replicas.values()}
        return len(heads) == 1
