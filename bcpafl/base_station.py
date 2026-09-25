"""Base station: global aggregation (Eq. 19), evaluation and model commitment.

    M = (1 / N_chi) * sum_k N_chi_k * M_k                               (Eq. 19)

N_chi_k is the number of vehicle updates RSU k actually aggregated this
round, N_chi their total.  RSUs that aggregated nothing carry zero weight;
if no RSU aggregated anything, M is unchanged.  The resulting model is signed
once (GLOBAL_MODEL broadcast relayed by every RSU) and its SHA-256 hash is
committed to the blockchain for auditability.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .blockchain import BlockchainNetwork, TX_MODEL_COMMIT
from .compression import CompressionError, compress, decompress
from .crypto.certificateless import KeyPair, SecurityError, Verifier
from .models import SharedModel, evaluate, get_vector, set_vector
from .rsu import MSG_CLUSTER_UPDATE
from .secure import EnvelopeOpener, sign_broadcast
from .vehicle import MSG_GLOBAL_MODEL
from .wire_codec import WireCodecError, decode_message


class BaseStation:
    bs_id = "BS"

    def __init__(self, keypair: KeyPair, chain: BlockchainNetwork, test_set: TensorDataset,
                 attack_set: Optional[TensorDataset], class_weights: torch.Tensor,
                 seed: int) -> None:
        self.keypair = keypair
        self.chain = chain
        self.test_set = test_set
        self.attack_set = attack_set
        self.class_weights = class_weights
        self.verifier = Verifier(keypair.P_pub)
        self.opener = EnvelopeOpener()
        # VANET-style instrumentation (additive; ignored by the core protocol).
        self.vanet_ms: Dict[str, float] = {}
        self._broadcast_sizes: Tuple[int, int] = (0, 0)
        torch.manual_seed(seed)
        self.model = SharedModel()
        self.vector = get_vector(self.model)
        self.num_params = self.vector.size
        self.version = 0

    @property
    def ledger(self):
        return self.chain.ledger(self.bs_id)

    def global_model_message(self, round_num: int) -> Dict:
        from .wire_codec import encode_message as _encode
        payload = compress(self.vector, "none")
        _t0 = time.perf_counter()
        msg = sign_broadcast(self.keypair, self.bs_id, MSG_GLOBAL_MODEL, self.version, payload)
        self.vanet_ms["sign_ms"] = self.vanet_ms.get("sign_ms", 0.0) + (
            time.perf_counter() - _t0) * 1000.0
        self._broadcast_sizes = (len(_encode(msg)), len(payload))
        return msg

    def evaluate(self) -> Dict[str, float]:
        set_vector(self.model, self.vector)
        result = evaluate(self.model, self.test_set, self.class_weights)
        if self.attack_set is not None:
            attack = evaluate(self.model, self.attack_set, self.class_weights)
            result["attack_accuracy"] = attack["accuracy"]
            result["attack_macro_f1"] = attack["macro_f1"]
        return result

    def aggregate(self, round_num: int, envelopes: Sequence[Dict]) -> Dict[str, object]:
        """Verify every CLUSTER_UPDATE, then apply Eq. (19)."""
        _sign_ms = self.vanet_ms.get("sign_ms", 0.0)
        self.vanet_ms = {"sign_ms": _sign_ms}
        opened: List[Tuple[str, np.ndarray, int, tuple]] = []
        rejected = 0
        for env in envelopes:
            sender = env.get("sender")
            try:
                _crypto: Dict[str, float] = {}
                payload, sig, pk, signed = self.opener.open(
                    env, MSG_CLUSTER_UPDATE, self.ledger.infra_public_key(sender), self.bs_id,
                    self.keypair, timings=_crypto)
                self.vanet_ms["decrypt_ms"] = self.vanet_ms.get("decrypt_ms", 0.0) + \
                    _crypto.get("decrypt_ms", 0.0)
                body = decode_message(payload)
                if body.get("base_version") != self.version or body.get("round") != round_num:
                    raise ValueError("cluster update for a different model version")
                vector = decompress(body["model"], self.num_params)
                n_k = int(body["n_accepted"])
                if n_k < 0:
                    raise ValueError("negative participant count")
            except (SecurityError, WireCodecError, CompressionError, ValueError, KeyError,
                    TypeError):
                rejected += 1
                continue
            opened.append((sender, vector, n_k, (signed, sig, pk)))
        _t0 = time.perf_counter()
        validity = self.verifier.verify_many([o[3] for o in opened])
        _batch_s = time.perf_counter() - _t0
        self.vanet_ms["batch_receiver_ms"] = self.vanet_ms.get(
            "batch_receiver_ms", 0.0) + _batch_s * 1000.0
        self.vanet_ms["batch_senders"] = [s for s, _, _, _ in opened]
        self.vanet_ms["batch_seconds"] = _batch_s
        clusters = [(s, v, n) for (s, v, n, _), ok in zip(opened, validity) if ok]
        rejected += sum(1 for ok in validity if not ok)
        n_total = sum(n for _, _, n in clusters)
        if n_total > 0:
            self.vector = sum(n * v for _, v, n in clusters) / n_total      # Eq. (19)
            self.vector = self.vector.astype(np.float32)
        self.version = round_num
        digest = hashlib.sha256(compress(self.vector, "none")).hexdigest()
        self.chain.submit(self.bs_id, TX_MODEL_COMMIT,
                          {"round": round_num, "model_sha256": digest,
                           "participants": n_total,
                           "clusters": {s: n for s, _, n in clusters}})
        return {"participants": n_total, "clusters": {s: n for s, _, n in clusters},
                "rejected_cluster_updates": rejected, "model_sha256": digest}
