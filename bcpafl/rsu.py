"""Road-Side Unit: authentication, POMDP control and adaptive aggregation.

Per round the RSU (Fig. 2, Algorithm 1):

* broadcasts ROUND_START (fresh nonce + BS-signed global model M);
* authenticates beacons against its own blockchain replica -- pseudonym
  registered, fresh, not revoked, key matches the ledger, signature valid
  (batch-verified), nonce fresh -- and rejects everything else;
* turns authenticated beacons into POMDP observations (Eq. 5), updates the
  belief (Eqs. 6-7), picks the action (Eq. 9) and selects vehicles (Eqs. 8,
  10, 11, 17, 18);
* after local training, batch-verifies uploads, screens anomalies, computes
  epsilon_i (Eq. 13) and aggregates M_k^(t+1) = M_k^t + sum eps_i dM_i
  (Eq. 12), evaluates the reward (Eqs. 14-15) and updates the controller;
* posts trust feedback to the ledger and forwards M_k to the base station.
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .blockchain import BlockchainNetwork, TX_TRUST_FEEDBACK
from .compression import CompressionError, compress, decompress
from .config import SimulationConfig
from .crypto.certificateless import KeyPair, PublicKey, SecurityError, Verifier
from .models import SharedModel, evaluate, set_vector
from .pomdp.belief import JointBelief
from .pomdp.controller import Observation, Plan, POMDPController
from .secure import EnvelopeOpener, seal, sign_broadcast
from .trust import (NEGATIVE_ANOMALY, NEGATIVE_DROPOUT, POSITIVE_ACCEPTED, AnomalyScreen,
                    ledger_trust)
from .vehicle import MSG_BEACON, MSG_LOCAL_UPDATE, MSG_ROUND_START, MSG_TRAIN_CONFIG
from .wire_codec import WireCodecError, decode_message, encode_message

MSG_CLUSTER_UPDATE = "CLUSTER_UPDATE"
MSG_BELIEF_REQUEST = "BELIEF_REQUEST"
MSG_BELIEF_RESPONSE = "BELIEF_RESPONSE"


@dataclass
class RoundLog:
    round: int
    rsu: str
    beacons: int = 0
    authenticated: int = 0
    auth_rejections: Dict[str, int] = field(default_factory=dict)
    belief_handovers: int = 0
    candidates: int = 0
    selected: int = 0
    delivered: int = 0
    dropouts: Dict[str, int] = field(default_factory=dict)
    anomalies: int = 0
    accepted: int = 0
    action: Dict[str, object] = field(default_factory=dict)
    reward: float = 0.0
    reward_terms: Dict[str, float] = field(default_factory=dict)
    val_accuracy_before: float = 0.0
    val_accuracy_after: float = 0.0
    aggregation_step: float = 1.0
    mean_q_selected: float = 0.0
    mean_psi_selected: float = 0.0
    brier: float = float("nan")
    upload_bytes: float = 0.0
    compute_seconds: float = 0.0
    xi: List[float] = field(default_factory=list)
    epsilon: float = 0.0
    malicious_selected: int = 0
    malicious_accepted: int = 0

    def reject(self, reason: str) -> None:
        self.auth_rejections[reason] = self.auth_rejections.get(reason, 0) + 1

    def drop(self, reason: str) -> None:
        self.dropouts[reason] = self.dropouts.get(reason, 0) + 1


class RSU:
    def __init__(self, rsu_id: str, xy: Tuple[float, float], cfg: SimulationConfig,
                 keypair: KeyPair, chain: BlockchainNetwork, validation: TensorDataset,
                 class_weights: torch.Tensor, num_params: int, rng: np.random.Generator) -> None:
        self.rsu_id = rsu_id
        self.xy = xy
        self.cfg = cfg
        self.keypair = keypair
        self.chain = chain
        self.validation = validation
        self.class_weights = class_weights
        self.num_params = num_params
        self.rng = rng
        self.verifier = Verifier(keypair.P_pub)
        self.opener = EnvelopeOpener()
        self.controller = POMDPController(rsu_id, xy, cfg, num_params, rng)
        self.screen = AnomalyScreen(cfg.trust_median_multiplier)
        self.model_vector: Optional[np.ndarray] = None
        self.model_version = -1
        self._eval_model = SharedModel()
        self._nonce = b""
        self._round = 0
        self._authenticated: Dict[str, Dict] = {}
        self._plan: Optional[Plan] = None
        self.log: Optional[RoundLog] = None
        self._accepted_count = 0
        # VANET-style instrumentation (additive; ignored by the core protocol).
        self.vanet_ms: Dict[str, float] = {}
        self._batch_records: List[Tuple[str, List[str], float]] = []
        self._config_sizes: Dict[str, Tuple[int, int]] = {}

    @property
    def ledger(self):
        return self.chain.ledger(self.rsu_id)

    # ------------------------------------------------------------------
    # Round start
    # ------------------------------------------------------------------
    def set_global(self, vector: np.ndarray, version: int) -> None:
        self.model_vector = vector.astype(np.float32).copy()
        self.model_version = version

    def start_round(self, round_num: int, global_model_msg: Dict) -> Dict:
        self._round = round_num
        self._nonce = secrets.token_bytes(16)
        self._authenticated = {}
        self._plan = None
        self._accepted_count = 0
        self.vanet_ms = {}
        self._batch_records = []
        self._config_sizes = {}
        self.log = RoundLog(round_num, self.rsu_id)
        payload = encode_message({"round": round_num, "nonce": self._nonce,
                                  "global_model": encode_message(global_model_msg)})
        _t0 = time.perf_counter()
        msg = sign_broadcast(self.keypair, self.rsu_id, MSG_ROUND_START, round_num, payload)
        self.vanet_ms["sign_ms"] = self.vanet_ms.get("sign_ms", 0.0) + (
            time.perf_counter() - _t0) * 1000.0
        self.vanet_ms["start_wire_bytes"] = float(len(encode_message(msg)))
        self.vanet_ms["start_app_bytes"] = float(len(payload))
        return msg

    # ------------------------------------------------------------------
    # Authentication (Sec. III-A: blockchain-managed pseudonyms)
    # ------------------------------------------------------------------
    def authenticate(self, beacons: Sequence[Dict]) -> List[str]:
        log = self.log
        opened = []
        for env in beacons:
            log.beacons += 1
            pid = env.get("sender") if isinstance(env, dict) else None
            if not isinstance(pid, str):
                log.reject("malformed")
                continue
            ok, reason = self.ledger.pseudonym_status(pid, self._round)
            if not ok:
                log.reject(reason)
                continue
            if pid in self._authenticated:
                log.reject("duplicate")
                continue
            try:
                _crypto: Dict[str, float] = {}
                payload, sig, pk, signed = self.opener.open(
                    env, MSG_BEACON, self.ledger.pseudonym_public_key(pid), self.rsu_id,
                    self.keypair, timings=_crypto)
                self.vanet_ms["decrypt_ms"] = self.vanet_ms.get("decrypt_ms", 0.0) + \
                    _crypto.get("decrypt_ms", 0.0)
                report = decode_message(payload)
            except (SecurityError, WireCodecError, ValueError) as exc:
                text = str(exc)
                log.reject("replay" if "replay" in text else
                           "key_mismatch" if "ledger-registered" in text else
                           "wrong_recipient" if "recipient" in text else "decryption")
                continue
            opened.append((pid, report, (signed, sig, pk)))
        _t0 = time.perf_counter()
        validity = self.verifier.verify_many([item[2] for item in opened])
        _batch_s = time.perf_counter() - _t0
        if opened:
            self._batch_records.append(
                (self.rsu_id, [pid for pid, _, _ in opened], _batch_s))
            self.vanet_ms["batch_receiver_ms"] = self.vanet_ms.get(
                "batch_receiver_ms", 0.0) + _batch_s * 1000.0
        accepted = []
        for (pid, report, _), valid in zip(opened, validity):
            if not valid:
                log.reject("bad_signature")
            elif report.get("nonce") != self._nonce:
                log.reject("stale_nonce")
            elif report.get("model_version") != self.model_version:
                log.reject("stale_model")
            else:
                self._authenticated[pid] = report
                accepted.append(pid)
        log.authenticated = len(accepted)
        return accepted

    def observations(self, pids: Sequence[str]) -> Tuple[List[Observation], Dict[str, float]]:
        obs, priors = [], {}
        for pid in pids:
            r = self._authenticated[pid]
            record = self.ledger.pseudonyms[pid]
            trust = ledger_trust(record)
            priors[pid] = trust
            px, py = r["position"]
            obs.append(Observation(
                pseudonym_id=pid, trust=trust,
                utility_raw=math.sqrt(max(r["num_samples"], 1)) * float(r["loss_rms"]),
                uncertainty=float(r["entropy"]), position=(float(px), float(py)),
                velocity=(float(r["velocity"][0]), float(r["velocity"][1])),
                num_samples=int(r["num_samples"]), compute_rate=float(r["compute_rate"]),
                distance_m=min(math.hypot(px - self.xy[0], py - self.xy[1]),
                               self.cfg.rsu_range_m)))
        return obs, priors

    # ------------------------------------------------------------------
    # I2I belief handover between RSUs
    # ------------------------------------------------------------------
    def handle_belief_request(self, message: Dict) -> Dict:
        pid = message.get("pseudonym_id")
        belief = self.controller.export_belief(pid) if isinstance(pid, str) else None
        return {"type": MSG_BELIEF_RESPONSE, "sender": self.rsu_id, "round": self._round,
                "pseudonym_id": pid, "belief": None if belief is None else belief.to_wire()}

    def import_belief(self, pid: str, wire: Optional[Dict]) -> bool:
        if wire is None:
            return False
        try:
            self.controller.import_belief(pid, JointBelief.from_wire(wire))
        except (KeyError, ValueError, TypeError):
            return False
        self.log.belief_handovers += 1
        return True

    def knows_belief(self, pid: str) -> bool:
        return pid in self.controller.beliefs

    # ------------------------------------------------------------------
    # POMDP planning and configuration broadcast (Algorithm 1, lines 3-9)
    # ------------------------------------------------------------------
    def plan(self, pids: Sequence[str], total_rounds: int) -> Plan:
        obs, priors = self.observations(pids)
        self.controller.observe(obs, priors)
        plan = self.controller.plan(pids, self._round, total_rounds, self.cfg.selection)
        self._plan = plan
        log = self.log
        log.candidates = len(pids)
        log.selected = len(plan.selected)
        log.action = plan.action.as_dict()
        if plan.selected:
            log.mean_q_selected = float(np.mean([c.q_avail for c in plan.selected]))
            log.mean_psi_selected = float(np.mean([c.psi for c in plan.selected]))
        return plan

    def build_configs(self, deadline_s: float) -> Dict[str, Dict]:
        configs = {}
        for c in self._plan.selected:
            body = encode_message({**self._plan.action.as_dict(),
                                   "model_version": self.model_version,
                                   "deadline_s": deadline_s, "round": self._round})
            _crypto: Dict[str, float] = {}
            env = seal(self.keypair, self.rsu_id, c.pseudonym_id,
                       self.ledger.pseudonym_public_key(c.pseudonym_id),
                       MSG_TRAIN_CONFIG, self._round, body, timings=_crypto)
            self.vanet_ms["sign_ms"] = self.vanet_ms.get("sign_ms", 0.0) + \
                _crypto.get("sign_ms", 0.0)
            self.vanet_ms["encrypt_ms"] = self.vanet_ms.get("encrypt_ms", 0.0) + \
                _crypto.get("encrypt_ms", 0.0)
            self._config_sizes[c.pseudonym_id] = (len(encode_message(env)), len(body))
            configs[c.pseudonym_id] = env
        return configs

    # ------------------------------------------------------------------
    # Aggregation and learning (Algorithm 1, lines 15-18)
    # ------------------------------------------------------------------
    def _accuracy(self, vector: np.ndarray) -> float:
        set_vector(self._eval_model, vector)
        return evaluate(self._eval_model, self.validation, self.class_weights)["accuracy"]

    def _loss(self, vector: np.ndarray) -> float:
        set_vector(self._eval_model, vector)
        return evaluate(self._eval_model, self.validation, self.class_weights)["loss"]

    def aggregate(self, uploads: Dict[str, Dict], failures: Dict[str, str],
                  upload_bytes: Dict[str, float], compute_seconds: Dict[str, float],
                  malicious_pids: set, final_round: bool) -> None:
        log, plan, cfg = self.log, self._plan, self.cfg
        selected = {c.pseudonym_id: c for c in plan.selected}
        for pid, reason in failures.items():
            log.drop(reason)

        # Decrypt, then batch-verify every delivered update at once.
        opened = []
        for pid, env in uploads.items():
            if pid not in selected or env.get("sender") != pid:
                log.drop("unsolicited")
                continue
            try:
                _crypto: Dict[str, float] = {}
                payload, sig, pk, signed = self.opener.open(
                    env, MSG_LOCAL_UPDATE, self.ledger.pseudonym_public_key(pid), self.rsu_id,
                    self.keypair, timings=_crypto)
                self.vanet_ms["decrypt_ms"] = self.vanet_ms.get("decrypt_ms", 0.0) + \
                    _crypto.get("decrypt_ms", 0.0)
                body = decode_message(payload)
                if body.get("base_version") != self.model_version:
                    raise ValueError("update built on a stale model")
                delta = decompress(body["delta"], self.num_params)
            except (SecurityError, WireCodecError, CompressionError, ValueError, KeyError):
                log.drop("invalid_update")
                continue
            opened.append((pid, delta, body, (signed, sig, pk)))
        _t0 = time.perf_counter()
        validity = self.verifier.verify_many([o[3] for o in opened])
        _batch_s = time.perf_counter() - _t0
        if opened:
            self._batch_records.append(
                (self.rsu_id, [pid for pid, _, _, _ in opened], _batch_s))
            self.vanet_ms["batch_receiver_ms"] = self.vanet_ms.get(
                "batch_receiver_ms", 0.0) + _batch_s * 1000.0
        deltas, bodies = {}, {}
        for (pid, delta, body, _), ok in zip(opened, validity):
            if ok:
                deltas[pid], bodies[pid] = delta, body
            else:
                log.drop("bad_signature")
        log.delivered = len(deltas)

        base = self.model_vector
        screening = self.screen.screen(deltas, lambda d: self._loss(base + d), self._loss(base))
        log.anomalies = len(screening.rejected)
        accepted = screening.accepted
        log.accepted = len(accepted)
        self._accepted_count = len(accepted)
        log.malicious_selected = sum(1 for pid in selected if pid in malicious_pids)
        log.malicious_accepted = sum(1 for pid in accepted if pid in malicious_pids)

        # Eq. (13) (or the policy's alternative strategy D) and Eq. (12).
        before = self.model_vector
        acc_before = self._accuracy(before)
        divergence = 0.0
        aggregate_delta = np.zeros_like(before)
        if accepted:
            strategy = plan.action.aggregation
            if strategy == "score":
                raw = np.array([max(selected[p].psi, 1e-12) for p in accepted])
            elif strategy == "data":
                raw = np.array([float(bodies[p]["num_samples"]) for p in accepted])
            else:
                raw = np.ones(len(accepted))
            eps = raw / raw.sum()
            for e, pid in zip(eps, accepted):
                aggregate_delta += e * deltas[pid]
            agg_norm = float(np.linalg.norm(aggregate_delta)) + 1e-8
            divergence = float(min(sum(e * np.linalg.norm(deltas[p] - aggregate_delta)
                                       for e, p in zip(eps, accepted)) / agg_norm, 3.0))
            step = 1.0
            candidate = before + aggregate_delta                      # Eq. (12)
            if cfg.rsu_validation_guard:
                # Class-weighted validation loss (not accuracy, which would favour
                # collapsing onto the majority benign class).
                limit = self._loss(before) * (1.0 + cfg.rsu_guard_tolerance)
                while step > 0.2 and self._loss(candidate) > limit:
                    step /= 2
                    candidate = before + step * aggregate_delta
                if step <= 0.2:
                    step, candidate = 0.0, before
            log.aggregation_step = step
            self.model_vector = candidate.astype(np.float32)
        acc_after = self._accuracy(self.model_vector)
        log.val_accuracy_before, log.val_accuracy_after = acc_before, acc_after

        # Reward (Eqs. 14-15).
        sent = sum(upload_bytes.get(p, 0.0) for p in selected)
        comp = sum(compute_seconds.get(p, 0.0) for p in selected)
        terms = POMDPController.reward_terms(acc_after - acc_before, sent, comp, divergence,
                                             [c.q_avail for c in plan.selected], cfg)
        log.reward = terms.total(cfg.reward_weights)
        log.reward_terms = {"delta_accuracy": terms.delta_accuracy, "comm": terms.comm_cost,
                            "comp": terms.comp_cost, "divergence": terms.divergence,
                            "dropout": terms.dropout_cost}
        log.upload_bytes, log.compute_seconds = sent, comp

        # Per-vehicle outcomes -> xi learning, availability calibration, trust feedback.
        outcomes, feedback = {}, []
        agg_norm = float(np.linalg.norm(aggregate_delta))
        brier = []
        for pid, cand in selected.items():
            success = pid in deltas
            brier.append((cand.q_avail - (1.0 if success else 0.0)) ** 2)
            if pid in accepted:
                cos = float(deltas[pid] @ aggregate_delta /
                            (np.linalg.norm(deltas[pid]) * agg_norm + 1e-12))
                outcomes[pid] = 0.5 + 0.5 * max(cos, 0.0)
                feedback.append({"pseudonym_id": pid, "positive": POSITIVE_ACCEPTED,
                                 "negative": 0.0, "anomalies": 0})
            elif pid in screening.rejected:
                outcomes[pid] = -2.0
                feedback.append({"pseudonym_id": pid, "positive": 0.0,
                                 "negative": NEGATIVE_ANOMALY, "anomalies": 1})
            else:
                outcomes[pid] = -1.0
                feedback.append({"pseudonym_id": pid, "positive": 0.0,
                                 "negative": NEGATIVE_DROPOUT, "anomalies": 0})
        log.brier = float(np.mean(brier)) if brier else float("nan")
        self.controller.finish_round(plan, outcomes, log.reward, final_round)
        log.xi = [float(x) for x in self.controller.score.xi]
        log.epsilon = float(self.controller.policy.epsilon)
        if feedback:
            self.chain.submit(self.rsu_id, TX_TRUST_FEEDBACK,
                              {"round": self._round, "entries": feedback})

    def finish_without_candidates(self, final_round: bool) -> None:
        """No authenticated vehicle this round: keep M_k, record an empty round."""
        self._accepted_count = 0
        if final_round:
            self.controller.policy.bootstrap(None)

    def cluster_update(self, bs_id: str, bs_pk: PublicKey) -> Dict:
        body = encode_message({"round": self._round, "n_accepted": self._accepted_count,
                               "model": compress(self.model_vector, "none"),
                               "base_version": self.model_version})
        _crypto: Dict[str, float] = {}
        env = seal(self.keypair, self.rsu_id, bs_id, bs_pk, MSG_CLUSTER_UPDATE, self._round,
                   body, timings=_crypto)
        self.vanet_ms["sign_ms"] = self.vanet_ms.get("sign_ms", 0.0) + \
            _crypto.get("sign_ms", 0.0)
        self.vanet_ms["encrypt_ms"] = self.vanet_ms.get("encrypt_ms", 0.0) + \
            _crypto.get("encrypt_ms", 0.0)
        self.vanet_ms["cluster_wire_bytes"] = float(len(encode_message(env)))
        self.vanet_ms["cluster_app_bytes"] = float(len(body))
        return env
