"""Vehicle (OBU) node: credentials, observations, local training, protected upload.

Per round (Fig. 2, Algorithm 1 lines 9-13) a vehicle in RSU coverage:

1. verifies the RSU's signed ROUND_START broadcast and the base-station
   signature on the global model M inside it, and loads M;
2. picks a random valid pseudonym and sends an authenticated, encrypted
   beacon with its observation report (noisy GPS kinematics, data size,
   compute rate, loss/entropy of M on local data);
3. if selected, receives the training configuration (f, Omega), trains M_i
   locally for f epochs -- Deep Mutual Learning with its private model when
   enabled -- computes Delta M_i = M_i - M, compresses it with error
   feedback, and uploads it signed and encrypted.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from .compression import ErrorFeedback, decompress
from .config import SimulationConfig
from .crypto.certificateless import PublicKey, SecurityError, Verifier
from .identity import Pseudonym, VehicleWallet
from .mobility import World
from .models import (PRIVATE_ARCHITECTURES, SharedModel, dml_kl, get_vector, local_statistics,
                     make_loader, set_vector)
from .privacy import epsilon_after
from .secure import EnvelopeOpener, broadcast_check_input, seal
from .wire_codec import decode_message, encode_message

MSG_ROUND_START = "ROUND_START"
MSG_GLOBAL_MODEL = "GLOBAL_MODEL"
MSG_BEACON = "AUTH_BEACON"
MSG_TRAIN_CONFIG = "TRAIN_CONFIG"
MSG_LOCAL_UPDATE = "LOCAL_UPDATE"


class Vehicle:
    def __init__(self, real_id: str, index: int, cfg: SimulationConfig, wallet: VehicleWallet,
                 train_data: TensorDataset, val_data: TensorDataset, class_weights: torch.Tensor,
                 p_pub, malicious: bool, rng: np.random.Generator) -> None:
        self.real_id = real_id
        self.index = index
        self.cfg = cfg
        self.wallet = wallet
        self.train_data = train_data
        self.val_data = val_data
        self.class_weights = class_weights
        self.malicious = malicious
        self.rng = rng
        self.verifier = Verifier(p_pub)
        self.opener = EnvelopeOpener()
        self.compute_rate = float(rng.uniform(*cfg.compute_samples_per_s))
        self.disconnect_prob = float(rng.uniform(*cfg.random_disconnect_prob))
        self.model = SharedModel()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.error_feedback = ErrorFeedback(self.num_params)
        self.private_model = None
        if cfg.use_private_models:
            torch.manual_seed(cfg.seed * 1000 + index)
            self.private_model = PRIVATE_ARCHITECTURES[index % len(PRIVATE_ARCHITECTURES)]()
            self.private_optimizer = torch.optim.Adam(self.private_model.parameters(),
                                                      lr=cfg.learning_rate)
        self.global_vector: Optional[np.ndarray] = None
        self.model_version = -1
        self.round_context: Dict[str, Any] = {}
        self.pseudonym: Optional[Pseudonym] = None
        self._loader_seed = cfg.seed * 7919 + index
        self.epsilon_spent = 0.0
        self._dp_steps = 0

    @property
    def num_samples(self) -> int:
        return len(self.train_data)

    # ------------------------------------------------------------------
    # Round start: verify the RSU broadcast and the BS-signed global model
    # ------------------------------------------------------------------
    def receive_round_start(self, message: Dict[str, Any], rsu_pk: Optional[PublicKey],
                            bs_pk: Optional[PublicKey]) -> bool:
        try:
            payload, sig, pk, signed = broadcast_check_input(message, MSG_ROUND_START, rsu_pk)
            inner = decode_message(payload)
            model_msg = decode_message(inner["global_model"])
            m_payload, m_sig, m_pk, m_signed = broadcast_check_input(model_msg, MSG_GLOBAL_MODEL,
                                                                     bs_pk)
        except (SecurityError, KeyError, TypeError, ValueError):
            return False
        _vt0 = time.perf_counter()
        verified = self.verifier.verify_many([(signed, sig, pk), (m_signed, m_sig, m_pk)])
        self.round_context["rs_verify_ms"] = (time.perf_counter() - _vt0) * 1000.0
        if not all(verified):
            return False
        if inner.get("round") != message.get("round"):
            return False
        vector = decompress(m_payload, self.num_params)
        self.global_vector = vector
        self.model_version = int(model_msg["round"])
        set_vector(self.model, vector)
        self.round_context = {"round": int(message["round"]), "rsu": message["sender"],
                              "nonce": inner["nonce"]}
        return True

    # ------------------------------------------------------------------
    # Beacon / authentication (Algorithm 1, line 2)
    # ------------------------------------------------------------------
    def build_beacon(self, world: World, rsu_id: str, rsu_pk: PublicKey) -> Optional[Dict]:
        round_num = self.round_context.get("round")
        if round_num is None or self.round_context.get("rsu") != rsu_id:
            return None
        self.wallet.prune(round_num)
        self.pseudonym = self.wallet.select(round_num, self.rng)
        if self.pseudonym is None:
            return None
        state = world.vehicles[self.real_id]
        vx, vy = state.velocity
        noise = self.cfg.gps_noise_m
        stats = local_statistics(self.model, self.train_data, self.class_weights)
        rn = self.cfg.report_noise_std
        report = {
            "nonce": self.round_context["nonce"],
            "position": [state.x + float(self.rng.normal(0, noise)),
                         state.y + float(self.rng.normal(0, noise))],
            "velocity": [vx + float(self.rng.normal(0, 0.1 * noise / 10)),
                         vy + float(self.rng.normal(0, 0.1 * noise / 10))],
            "num_samples": self.num_samples,
            "compute_rate": self.compute_rate,
            "loss_rms": max(0.0, stats["loss_rms"] * (1 + float(self.rng.normal(0, rn)))),
            "entropy": float(np.clip(stats["entropy"] + self.rng.normal(0, rn), 0.0, 1.0)),
            "model_version": self.model_version,
        }
        app_bytes = encode_message(report)
        crypto_timings: Dict[str, float] = {}
        env = seal(self.pseudonym.keypair, self.pseudonym.pseudonym_id, rsu_id, rsu_pk,
                   MSG_BEACON, round_num, app_bytes, timings=crypto_timings)
        self.round_context["beacon_sizes"] = (len(encode_message(env)), len(app_bytes))
        self.round_context["beacon_crypto_ms"] = crypto_timings
        return env

    def receive_config(self, envelope: Dict, rsu_pk: Optional[PublicKey]) -> Optional[Dict]:
        if self.pseudonym is None:
            return None
        try:
            crypto_timings: Dict[str, float] = {}
            payload, sig, pk, signed = self.opener.open(
                envelope, MSG_TRAIN_CONFIG, rsu_pk, self.pseudonym.pseudonym_id,
                self.pseudonym.keypair, timings=crypto_timings)
        except SecurityError:
            return None
        _vt0 = time.perf_counter()
        ok = self.verifier.verify(signed, sig, pk)
        self.round_context["config_crypto_ms"] = {
            "decrypt_ms": crypto_timings.get("decrypt_ms", 0.0),
            "verify_ms": (time.perf_counter() - _vt0) * 1000.0,
        }
        if not ok:
            return None
        config = decode_message(payload)
        if config.get("model_version") != self.model_version:
            return None
        return config

    # ------------------------------------------------------------------
    # Local training (Algorithm 1, line 11; Eq. 1)
    # ------------------------------------------------------------------
    def _dp_step(self, x: torch.Tensor, y: torch.Tensor, soft: Optional[torch.Tensor],
                 optimizer: torch.optim.Optimizer) -> None:
        """Per-sample clipped Gaussian DP-SGD on the shared model (ProxyFL v1)."""
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        cfg = self.cfg

        def loss_fn(p, xi, yi, si):
            out = torch.func.functional_call(self.model, p, (xi.unsqueeze(0),))
            loss = F.cross_entropy(out, yi.unsqueeze(0), weight=self.class_weights)
            if si is not None:
                loss = (1 - cfg.dml_beta) * loss + cfg.dml_beta * dml_kl(
                    out, si.unsqueeze(0), cfg.dml_temperature)
            return loss

        if soft is None:
            grads = torch.func.vmap(torch.func.grad(lambda p, a, b: loss_fn(p, a, b, None)),
                                    in_dims=(None, 0, 0))(params, x, y)
        else:
            grads = torch.func.vmap(torch.func.grad(loss_fn), in_dims=(None, 0, 0, 0))(
                params, x, y, soft)
        b = x.shape[0]
        flat = torch.cat([g.reshape(b, -1) for g in grads.values()], dim=1)
        factors = torch.clamp(cfg.dp_clip_norm / (flat.norm(dim=1) + 1e-6), max=1.0)
        optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            g = grads[name] * factors.view([b] + [1] * (grads[name].dim() - 1))
            noise = torch.randn_like(param) * cfg.dp_noise_multiplier * cfg.dp_clip_norm
            param.grad = (g.sum(dim=0) + noise) / b
        optimizer.step()
        self._dp_steps += 1

    def train(self, local_epochs: int) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """Returns (Delta M_i, simulated compute seconds, training stats)."""
        if self.global_vector is None:
            raise RuntimeError("vehicle has no global model to train from")
        cfg = self.cfg
        set_vector(self.model, self.global_vector)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self._loader_seed += 1
        loader = make_loader(self.train_data, cfg.batch_size, self._loader_seed)
        total_loss, batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in loader:
                self.model.train()
                soft_private = None
                if self.private_model is not None:
                    self.private_model.train()
                    with torch.no_grad():
                        shared_soft = F.softmax(self.model(x) / cfg.dml_temperature, dim=1)
                    priv_out = self.private_model(x)
                    priv_loss = ((1 - cfg.dml_alpha) * F.cross_entropy(
                        priv_out, y, weight=self.class_weights)
                        + cfg.dml_alpha * dml_kl(priv_out, shared_soft, cfg.dml_temperature))
                    self.private_optimizer.zero_grad()
                    priv_loss.backward()
                    self.private_optimizer.step()
                    with torch.no_grad():
                        soft_private = F.softmax(self.private_model(x) / cfg.dml_temperature,
                                                 dim=1)
                if cfg.dp_noise_multiplier > 0:
                    self._dp_step(x, y, soft_private, optimizer)
                    with torch.no_grad():
                        loss = F.cross_entropy(self.model(x), y, weight=self.class_weights)
                else:
                    out = self.model(x)
                    loss = F.cross_entropy(out, y, weight=self.class_weights)  # Eq. (1)
                    if soft_private is not None:
                        loss = (1 - cfg.dml_beta) * loss + cfg.dml_beta * dml_kl(
                            out, soft_private, cfg.dml_temperature)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.item())
                batches += 1
        delta = get_vector(self.model) - self.global_vector
        if self.malicious:
            # Model poisoning: a scaled, sign-flipped update.
            delta = -cfg.malicious_scale * delta
        jitter = float(self.rng.lognormal(0.0, 0.1))
        compute_s = local_epochs * self.num_samples / self.compute_rate * jitter
        stats = {"train_loss": total_loss / max(batches, 1)}
        if self.private_model is not None:
            with torch.no_grad():
                self.private_model.eval()
                x, y = self.val_data.tensors
                stats["private_val_accuracy"] = float(
                    (self.private_model(x).argmax(1) == y).float().mean())
        if cfg.dp_noise_multiplier > 0:
            self.epsilon_spent = epsilon_after(
                self._dp_steps, cfg.dp_noise_multiplier,
                min(cfg.batch_size / max(self.num_samples, 1), 1.0), cfg.dp_delta)
            stats["epsilon"] = self.epsilon_spent
        return delta.astype(np.float32), compute_s, stats

    def build_upload(self, delta: np.ndarray, compression: str, rsu_id: str,
                     rsu_pk: PublicKey, local_epochs: int) -> Dict:
        round_num = self.round_context["round"]
        payload = self.error_feedback.compress(delta, compression,
                                               topk_fraction=self.cfg.topk_fraction, rng=self.rng)
        body = encode_message({"delta": payload, "num_samples": self.num_samples,
                               "local_epochs": local_epochs, "base_version": self.model_version,
                               "compression": compression})
        crypto_timings: Dict[str, float] = {}
        env = seal(self.pseudonym.keypair, self.pseudonym.pseudonym_id, rsu_id, rsu_pk,
                   MSG_LOCAL_UPDATE, round_num, body, timings=crypto_timings)
        self.round_context["upload_sizes"] = (len(encode_message(env)), len(body))
        self.round_context["upload_crypto_ms"] = crypto_timings
        return env

    def disconnects_this_round(self) -> Tuple[bool, float]:
        """Temporary loss of connectivity: (happens, time offset in the round)."""
        happens = bool(self.rng.random() < self.disconnect_prob)
        return happens, float(self.rng.uniform(0, self.cfg.round_duration_s))
