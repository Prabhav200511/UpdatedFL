"""End-to-end BC-PAFL simulation (Algorithm 1 + Fig. 2).

One call to :meth:`Simulation.run_round` executes a full federated round in
simulated time::

    pseudonym maintenance (TA -> blockchain block)
    for every RSU k in parallel:
        ROUND_START broadcast (nonce + BS-signed M)            -- Fig. 2 "broadcast M"
        vehicle beacons -> authentication on the ledger       -- Sec. III-A
        observations -> belief update (Eqs. 5-7)              -- lines 2-3
        availability + score (Eqs. 8, 10)                     -- lines 4-7
        action + selection (Eqs. 9, 11, 17, 18)               -- lines 8-9
        TRAIN_CONFIG -> local training + compression           -- lines 10-13
    uploads processed in simulated-time order: vehicles that left coverage,
      lost connectivity, lost frames or missed the deadline drop out
    RSU aggregation (Eqs. 12-13), reward (Eqs. 14-15), learning -- lines 15-18
    base-station aggregation (Eq. 19) and evaluation          -- line 20
    trust feedback + model commitment (blockchain block), TA tracing/revocation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import channel
from .base_station import BaseStation
from .blockchain import ROLE_BS, ROLE_RSU, ROLE_TA, BlockchainNetwork
from .config import DATA_DIR, SimulationConfig
from .crypto.certificateless import KeyGenerationCenter, KeyPair
from .data import load_federated_data
from .identity import TrustedAuthority, VehicleWallet
from .mobility import World
from .models import evaluate
from .network import Network
from .rsu import MSG_BELIEF_REQUEST, RSU
from .secure import seal
from .vanet_aodv import AodvSettings, RoutingSimulator, TopologySnapshot
from .vanet_metrics import SERVER_NODE, VanetTracker
from .vehicle import MSG_BEACON, Vehicle
from .wire_codec import encode_message

V2V_RANGE_M = 350.0       # V2V adjacency radius for the modeled AODV overlay (v1 value)

CONTROL_PHASE_S = 1.0      # beacon + selection + configuration exchange


class Adversary:
    """Outsider attempting to join the FL candidate pool without valid credentials."""

    KINDS = ("unregistered", "key_mismatch", "replay")

    def __init__(self, p_pub_real, rng: np.random.Generator) -> None:
        rogue_kgc = KeyGenerationCenter()          # not the federation's KGC
        rogue_ta = TrustedAuthority(rogue_kgc)
        rogue_ta.enroll("intruder")
        aid = rogue_ta._generate_aid("intruder")
        self.keypair = KeyPair(aid, rogue_kgc.extract_partial_private_key(aid), rogue_kgc.P_pub)
        self.rng = rng
        self.captured: List[Dict] = []          # beacons overheard in earlier rounds
        self._staged: List[Dict] = []           # beacons overheard this round
        self.attempts: Dict[str, int] = {k: 0 for k in self.KINDS}

    def capture(self, envelope: Dict) -> None:
        """Eavesdrop on the air interface."""
        self._staged.append(dict(envelope))

    def end_round(self) -> None:
        self.captured = (self.captured + self._staged)[-20:]
        self._staged = []

    def forge(self, kind: str, rsu_id: str, rsu_pk, round_num: int,
              victim_pid: Optional[str]) -> Optional[Dict]:
        report = encode_message({"nonce": b"\x00" * 16, "position": [0.0, 0.0],
                                 "velocity": [0.0, 0.0], "num_samples": 5000,
                                 "compute_rate": 500.0, "loss_rms": 9.0, "entropy": 0.0,
                                 "model_version": 0})
        if kind == "unregistered":
            pid = self.keypair.public_key.fingerprint()
        elif kind == "key_mismatch" and victim_pid is not None:
            pid = victim_pid                         # claims someone else's pseudonym
        elif kind == "replay" and self.captured:
            # Prefer a beacon originally sent to this RSU: only the nonce cache /
            # freshness checks can then tell the copy from the original.
            same = [e for e in self.captured if e.get("recipient") == rsu_id] or self.captured
            self.attempts[kind] += 1
            return dict(same[int(self.rng.integers(len(same)))])
        else:
            return None
        self.attempts[kind] += 1
        return seal(self.keypair, pid, rsu_id, rsu_pk, MSG_BEACON, round_num, report)


class Simulation:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg.validate()
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.link = cfg.link_params()
        self.network = Network(self.link, np.random.default_rng(cfg.seed + 1))
        rsu_names = [name for name, _, _ in cfg.rsu_layout]
        self.data = load_federated_data(
            cfg.num_vehicles, rsu_names, seed=cfg.seed, dirichlet_alpha=cfg.dirichlet_alpha,
            max_samples_per_vehicle=cfg.max_samples_per_vehicle,
            rsu_validation_rows=cfg.rsu_validation_rows, data_dir=DATA_DIR)

        # --- Trust infrastructure: KGC, TA, infrastructure keys, blockchain -------
        t0 = time.perf_counter()
        self.kgc = KeyGenerationCenter()
        self.ta = TrustedAuthority(self.kgc)
        ta_kp = self.ta.issue_infrastructure_key(TrustedAuthority.ta_id)
        bs_kp = self.ta.issue_infrastructure_key(BaseStation.bs_id)
        rsu_kps = {name: self.ta.issue_infrastructure_key(name) for name in rsu_names}
        validators = {TrustedAuthority.ta_id: (ROLE_TA, ta_kp), BaseStation.bs_id: (ROLE_BS, bs_kp)}
        validators.update({name: (ROLE_RSU, kp) for name, kp in rsu_kps.items()})
        self.chain = BlockchainNetwork(self.kgc.P_pub, validators)
        self.ta.attach_chain(self.chain, ta_kp)
        self.setup_security_s = time.perf_counter() - t0

        # --- Mobility ---------------------------------------------------------
        self.world = World(cfg.rsu_layout, cfg.rsu_range_m, cfg.area_half_width_m,
                           cfg.speed_range_mps, cfg.heading_jitter_rad,
                           np.random.default_rng(cfg.seed + 2))

        # --- Vehicles -----------------------------------------------------------
        n_mal = int(round(cfg.malicious_fraction * cfg.num_vehicles))
        malicious = set(self.rng.choice(cfg.num_vehicles, size=n_mal, replace=False).tolist()) \
            if n_mal else set()
        self.vehicles: Dict[str, Vehicle] = {}
        for i in range(cfg.num_vehicles):
            vid = f"V{i:03d}"
            self.ta.enroll(vid)
            near = rsu_names[i % len(rsu_names)] if self.rng.random() < 0.8 else None
            self.world.spawn(vid, near)
            self.vehicles[vid] = Vehicle(
                vid, i, cfg, VehicleWallet(vid, self.kgc.P_pub), self.data.vehicle_datasets[i],
                self.data.vehicle_validation[i], self.data.class_weights, self.kgc.P_pub,
                i in malicious, np.random.default_rng(cfg.seed * 31 + i))
        self.malicious_ids = {vid for vid, v in self.vehicles.items() if v.malicious}

        # --- Base station and RSUs -----------------------------------------------
        self.bs = BaseStation(bs_kp, self.chain, self.data.global_test,
                              self.data.attack_benchmark, self.data.class_weights, cfg.seed)
        self.rsus: Dict[str, RSU] = {}
        for k, (name, x, y) in enumerate(cfg.rsu_layout):
            rsu = RSU(name, (x, y), cfg, rsu_kps[name], self.chain,
                      self.data.rsu_validation[name], self.data.class_weights,
                      self.bs.num_params, np.random.default_rng(cfg.seed * 97 + k))
            rsu.set_global(self.bs.vector, self.bs.version)
            self.rsus[name] = rsu

        self.adversary = Adversary(self.kgc.P_pub, np.random.default_rng(cfg.seed + 3))
        self.round_rows: List[Dict] = []
        self.rsu_rows: List[Dict] = []
        self.revocations: List[Tuple[int, str]] = []
        # VANET-style instrumentation (v1-compatible outputs).
        self.vanet = VanetTracker()
        self.vanet_quality: Dict[tuple, Dict[str, float]] = {}
        self.aodv = RoutingSimulator(
            settings=AodvSettings(), capacity=lambda d: channel.capacity_bps(d, self.link),
            seed=cfg.seed)
        self._provision_s: Dict[str, float] = {}
        initial = self.bs.evaluate()
        self.initial_metrics = initial
        self._log(f"[init] {cfg.num_vehicles} vehicles ({len(malicious)} malicious), "
                  f"{len(rsu_names)} RSUs, model {self.bs.num_params} params, "
                  f"initial test acc {initial['accuracy']:.3f}")

    # ------------------------------------------------------------------
    def _log(self, text: str) -> None:
        if self.cfg.verbose:
            print(text, flush=True)

    def _maintain_pseudonyms(self, round_num: int) -> int:
        issued = 0
        self._provision_s = {}
        for vid, vehicle in self.vehicles.items():
            vehicle.wallet.prune(round_num)
            if not vehicle.wallet.valid(round_num) and not self.ta.is_revoked(vid):
                _t0 = time.perf_counter()
                issued += len(self.ta.provision(vehicle.wallet, round_num,
                                                self.cfg.pseudonym_pool_size,
                                                self.cfg.pseudonym_lifetime_rounds))
                self._provision_s[vid] = time.perf_counter() - _t0
        return issued

    def _owner_of(self, pid: str) -> Optional[Vehicle]:
        for vehicle in self.vehicles.values():
            if vehicle.pseudonym is not None and vehicle.pseudonym.pseudonym_id == pid:
                return vehicle
        return None

    # ------------------------------------------------------------------
    def run_round(self, t: int) -> Dict:
        cfg = self.cfg
        wall0 = time.perf_counter()
        t0 = (t - 1) * cfg.round_duration_s
        deadline = t0 + cfg.round_duration_s
        self.world.advance_to(t0)

        # 1. Pseudonym maintenance + pending revocations -> one block.
        sec0 = time.perf_counter()
        issued = self._maintain_pseudonyms(t)
        self.chain.commit_pending(t)
        security_s = time.perf_counter() - sec0

        members = self.world.members()
        for vehicle in self.vehicles.values():
            vehicle.round_context = {}
            vehicle.pseudonym = None
        # Ground-truth positions at round start (frozen for the AODV overlay).
        vpos = {vid: (st.x, st.y) for vid, st in self.world.vehicles.items()}
        # VANET per-round accumulators: bytes[node] = [tx, rx], comm[node] = [tx_ms, rx_ms].
        vbytes: Dict[str, List[float]] = {}
        vcomm: Dict[str, List[float]] = {}

        def _bump(store: Dict[str, List[float]], node: str, tx: float, rx: float) -> None:
            entry = store.setdefault(node, [0.0, 0.0])
            entry[0] += tx
            entry[1] += rx

        vtrain: Dict[str, Dict] = {}
        va2r: Dict[str, float] = {}
        global_msg = self.bs.global_model_message(t)
        ref = self.chain.reference()
        bs_pk = ref.infra_public_key(BaseStation.bs_id)

        # 2. Per-RSU control phase.
        pending_events = []           # (t_done, rsu, pid, vehicle, envelope, epochs, compute)
        failures: Dict[str, Dict[str, str]] = {name: {} for name in self.rsus}
        upload_bytes: Dict[str, Dict[str, float]] = {name: {} for name in self.rsus}
        compute: Dict[str, Dict[str, float]] = {name: {} for name in self.rsus}
        train_wall = 0.0
        train_stats = []
        valid_pids = [pid for pid in ref.pseudonyms if ref.pseudonym_status(pid, t)[0]]
        for k, (name, rsu) in enumerate(self.rsus.items()):
            rsu_pk = ref.infra_public_key(name)
            start = rsu.start_round(t, global_msg)
            distances = {vid: self.world.distance(vid, name) for vid in members[name]}
            received = self.network.broadcast(start, distances)
            _bump(vbytes, name, len(encode_message(start)), 0.0)
            if received:
                _bump(vcomm, name, next(iter(received.values())).airtime_s * 1000.0, 0.0)
            beacons = []
            for vid, delivery in received.items():
                vehicle = self.vehicles[vid]
                if delivery.delivered:
                    _bump(vbytes, vid, 0.0, delivery.num_bytes)
                    _bump(vcomm, vid, 0.0, delivery.airtime_s * 1000.0)
                if not delivery.delivered or not vehicle.receive_round_start(
                        delivery.message, rsu_pk, bs_pk):
                    continue
                env = vehicle.build_beacon(self.world, name, rsu_pk)
                if env is None:
                    continue
                self.adversary.capture(env)
                d = self.network.v2i(env, distances[vid])
                _bump(vbytes, vid, d.num_bytes, 0.0)
                _bump(vcomm, vid, d.airtime_s * 1000.0, 0.0)
                if d.delivered:
                    _bump(vbytes, name, 0.0, d.num_bytes)
                    _bump(vcomm, name, 0.0, d.airtime_s * 1000.0)
                    beacons.append(d.message)
            # Adversarial join attempts (rejected by ledger checks / AEAD / nonces).
            for j in range(cfg.forged_auth_attempts_per_round):
                kind = Adversary.KINDS[(t + j + k) % len(Adversary.KINDS)]
                victim = (valid_pids[int(self.adversary.rng.integers(len(valid_pids)))]
                          if valid_pids else None)
                forged = self.adversary.forge(kind, name, rsu_pk, t, victim)
                if forged is not None:
                    beacons.append(forged)
            pids = rsu.authenticate(beacons)
            # I2I belief handover for pseudonyms another RSU has seen.
            for pid in pids:
                if rsu.knows_belief(pid):
                    continue
                for other_name, other in self.rsus.items():
                    if other_name == name or not other.knows_belief(pid):
                        continue
                    req = self.network.i2i({"type": MSG_BELIEF_REQUEST, "sender": name,
                                            "recipient": other_name, "round": t,
                                            "pseudonym_id": pid})
                    resp = self.network.i2i(other.handle_belief_request(req.message))
                    rsu.import_belief(pid, resp.message.get("belief"))
                    break
            if not pids:
                continue
            plan = rsu.plan(pids, cfg.rounds)
            configs = rsu.build_configs(cfg.round_duration_s)
            for cand in plan.selected:
                pid = cand.pseudonym_id
                vehicle = self._owner_of(pid)
                if vehicle is None:
                    failures[name][pid] = "unknown_owner"
                    continue
                dist = self.world.distance(vehicle.real_id, name)
                d = self.network.v2i(configs[pid], dist)
                _bump(vbytes, name, d.num_bytes, 0.0)
                _bump(vcomm, name, d.airtime_s * 1000.0, 0.0)
                config = vehicle.receive_config(d.message, rsu_pk) if d.delivered else None
                if d.delivered:
                    _bump(vbytes, vehicle.real_id, 0.0, d.num_bytes)
                    _bump(vcomm, vehicle.real_id, 0.0, d.airtime_s * 1000.0)
                if config is None:
                    failures[name][pid] = "config_lost"
                    continue
                epochs = int(config["local_epochs"])
                tw = time.perf_counter()
                delta, compute_s, stats = vehicle.train(epochs)
                train_s = time.perf_counter() - tw
                train_wall += train_s
                train_stats.append(stats)
                vtrain[vehicle.real_id] = {"stats": stats, "wall_s": train_s,
                                           "epochs": epochs, "rsu": name,
                                           "num_samples": vehicle.num_samples}
                envelope = vehicle.build_upload(delta, config["compression"], name, rsu_pk, epochs)
                compute[name][pid] = compute_s
                pending_events.append((t0 + CONTROL_PHASE_S + compute_s, name, pid, vehicle,
                                       envelope))

        # 3. Upload phase in simulated-time order (dropouts happen here).
        disconnects = {vid: v.disconnects_this_round() for vid, v in self.vehicles.items()}
        uploads: Dict[str, Dict[str, Dict]] = {name: {} for name in self.rsus}
        for t_done, name, pid, vehicle, envelope in sorted(pending_events, key=lambda e: e[0]):
            if t_done > deadline:
                failures[name][pid] = "deadline"
                continue
            self.world.advance_to(t_done)
            happens, offset = disconnects[vehicle.real_id]
            if happens and t0 + offset <= t_done:
                failures[name][pid] = "disconnected"
                continue
            if not self.world.in_range(vehicle.real_id, name):
                failures[name][pid] = "left_coverage"
                continue
            d = self.network.v2i(envelope, self.world.distance(vehicle.real_id, name))
            upload_bytes[name][pid] = d.num_bytes
            if d.delivered:
                _bump(vbytes, vehicle.real_id, d.num_bytes, 0.0)
                _bump(vbytes, name, 0.0, d.num_bytes)
                _bump(vcomm, vehicle.real_id, d.airtime_s * 1000.0, 0.0)
                _bump(vcomm, name, 0.0, d.airtime_s * 1000.0)
            if not d.delivered:
                failures[name][pid] = "channel"
            elif t_done + d.airtime_s > deadline:
                failures[name][pid] = "deadline"
            else:
                uploads[name][pid] = d.message
                va2r[vehicle.real_id] = (t_done + d.airtime_s
                                         - (t0 + CONTROL_PHASE_S)) * 1000.0
        self.world.advance_to(deadline)

        # 4. RSU aggregation and learning, then Eq. (19) at the base station.
        final = t == cfg.rounds
        malicious_pids = {v.pseudonym.pseudonym_id for v in self.vehicles.values()
                          if v.malicious and v.pseudonym is not None}
        cluster_msgs = []
        for name, rsu in self.rsus.items():
            if rsu._plan is None:
                rsu.finish_without_candidates(final)
            else:
                rsu.aggregate(uploads[name], failures[name], upload_bytes[name], compute[name],
                              malicious_pids, final)
            cup = rsu.cluster_update(BaseStation.bs_id, bs_pk)
            cd = self.network.i2i(cup)
            _bump(vbytes, name, cd.num_bytes, 0.0)
            _bump(vbytes, "BS", 0.0, cd.num_bytes)
            cluster_msgs.append(cd.message)
        bs_result = self.bs.aggregate(t, cluster_msgs)
        for rsu in self.rsus.values():
            rsu.set_global(self.bs.vector, self.bs.version)
        metrics = self.bs.evaluate()

        # 5. Feedback + model commitment -> block; TA traces and revokes.
        sec1 = time.perf_counter()
        self.chain.commit_pending(t)
        revoked = self.ta.process_feedback(self.chain.reference(), t, cfg.trust_forgetting,
                                           cfg.revoke_malicious_after)
        for real_id in revoked:
            self.revocations.append((t, real_id))
        self.adversary.end_round()
        security_s += time.perf_counter() - sec1

        # 6. Metrics.
        logs = [rsu.log for rsu in self.rsus.values() if rsu.log is not None]
        for log in logs:
            row = asdict(log)
            row["auth_rejections"] = json.dumps(row["auth_rejections"], sort_keys=True)
            row["dropouts"] = json.dumps(row["dropouts"], sort_keys=True)
            row["action"] = json.dumps(row["action"], sort_keys=True)
            row["reward_terms"] = json.dumps(row["reward_terms"], sort_keys=True)
            row["xi"] = json.dumps(row["xi"])
            self.rsu_rows.append(row)
        selected = sum(l.selected for l in logs)
        accepted = sum(l.accepted for l in logs)
        dropout_total = sum(sum(l.dropouts.values()) for l in logs)
        rejections: Dict[str, int] = {}
        for l in logs:
            for k, v in l.auth_rejections.items():
                rejections[k] = rejections.get(k, 0) + v
        briers = [l.brier for l in logs if np.isfinite(l.brier)]
        ref = self.chain.reference()
        row = {
            "round": t,
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_loss": metrics["loss"],
            "attack_accuracy": metrics.get("attack_accuracy", float("nan")),
            "vehicles_in_coverage": sum(len(v) for v in members.values()),
            "authenticated": sum(l.authenticated for l in logs),
            "auth_rejected": sum(rejections.values()),
            "auth_rejections": json.dumps(rejections, sort_keys=True),
            "selected": selected,
            "delivered": sum(l.delivered for l in logs),
            "accepted": accepted,
            "dropouts": dropout_total,
            "dropout_rate": dropout_total / selected if selected else 0.0,
            "anomalies": sum(l.anomalies for l in logs),
            "malicious_selected": sum(l.malicious_selected for l in logs),
            "malicious_accepted": sum(l.malicious_accepted for l in logs),
            "mean_reward": float(np.mean([l.reward for l in logs if l.selected])) if selected else 0.0,
            "availability_brier": float(np.mean(briers)) if briers else float("nan"),
            "upload_bytes": sum(l.upload_bytes for l in logs),
            "compute_seconds": sum(l.compute_seconds for l in logs),
            "bs_participants": bs_result["participants"],
            "pseudonyms_issued": issued,
            "revoked_identities": len(self.revocations),
            "chain_height": ref.height,
            "chain_valid": ref.verify_chain() if final else True,
            "replicas_consistent": self.chain.replicas_consistent(),
            "v2i_bytes": self.network.counters.get("v2i_bytes", 0.0),
            "broadcast_bytes": self.network.counters.get("broadcast_bytes", 0.0),
            "i2i_bytes": self.network.counters.get("i2i_bytes", 0.0),
            "train_wall_s": train_wall,
            "security_ledger_wall_s": security_s,
            "round_wall_s": time.perf_counter() - wall0,
            "model_sha256": bs_result["model_sha256"],
        }
        if train_stats and "private_val_accuracy" in train_stats[0]:
            row["private_val_accuracy"] = float(np.mean([s["private_val_accuracy"]
                                                         for s in train_stats]))
        self.round_rows.append(row)
        self._vanet_finish_round(t, t0, members, logs, metrics, bs_result, row["round_wall_s"],
                                 vbytes, vcomm, vtrain, va2r, vpos)
        self._log(f"[round {t:>3}] acc={row['test_accuracy']:.4f} f1={row['test_macro_f1']:.4f} "
                  f"cov={row['vehicles_in_coverage']} auth={row['authenticated']} "
                  f"rej={row['auth_rejected']} sel={selected} ok={accepted} "
                  f"drop={dropout_total} anom={row['anomalies']} "
                  f"revoked={len(self.revocations)} chain={ref.height} "
                  f"({row['round_wall_s']:.1f}s)")
        return row

    def run(self) -> Dict:
        for t in range(1, self.cfg.rounds + 1):
            self.run_round(t)
        return self.summary()

    # ------------------------------------------------------------------
    # VANET-style outputs (ProxyFL v1 compatible metrics + AODV overlay)
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_sizes(wire, app) -> Optional[tuple]:
        try:
            w = max(1, int(wire))
            a = max(0, min(int(app), w))
        except (TypeError, ValueError):
            return None
        return w, a

    @torch.no_grad()
    def _vanet_vehicle_quality(self, vehicle: Vehicle, stats: Dict) -> Dict[str, float]:
        """Post-train shared-model and private-model quality for one vehicle."""
        vehicle.model.eval()
        x, y = vehicle.train_data.tensors
        n = min(512, len(x))
        train_acc = float((vehicle.model(x[:n]).argmax(1) == y[:n]).float().mean())
        if "private_val_accuracy" in stats:
            priv_acc = float(stats["private_val_accuracy"])
        elif vehicle.private_model is not None:
            vehicle.private_model.eval()
            xv, yv = vehicle.val_data.tensors
            priv_acc = float((vehicle.private_model(xv).argmax(1) == yv).float().mean())
        else:
            priv_acc = float(evaluate(vehicle.model, vehicle.val_data,
                                      vehicle.class_weights)["accuracy"])
        return {"train_loss": float(stats.get("train_loss", 0.0)),
                "train_accuracy_pct": train_acc * 100.0,
                "private_test_accuracy_pct": priv_acc * 100.0,
                "epsilon": float(stats.get("epsilon", 0.0)),
                "delta": float(self.cfg.dp_delta)}

    def _vanet_snapshot(self, vpos: Dict[str, tuple]) -> TopologySnapshot:
        rsu_pos = {name: self.world.rsus[name] for name in self.rsus}
        nodes = list(self.vehicles) + list(self.rsus)
        edges = []
        vids = list(self.vehicles)
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                d = float(((vpos[a][0] - vpos[b][0]) ** 2
                           + (vpos[a][1] - vpos[b][1]) ** 2) ** 0.5)
                if d <= V2V_RANGE_M:
                    edges.append((a, b, d))
        for vid in vids:
            for name, (cx, cy) in rsu_pos.items():
                d = float(((vpos[vid][0] - cx) ** 2 + (vpos[vid][1] - cy) ** 2) ** 0.5)
                if d <= self.cfg.rsu_range_m:
                    edges.append((vid, name, d))
        return TopologySnapshot.from_edges(nodes, edges)

    def _vanet_submit(self, t: int, arrival: float, src: str, dst: str,
                      wire, app) -> None:
        sizes = self._clamp_sizes(wire, app)
        if sizes is None or src == dst:
            return
        try:
            delivery = self.aodv.submit(src, dst, sizes[0], sizes[1], t,
                                        self._vanet_current_snapshot,
                                        arrival_time=arrival)
        except ValueError:
            return
        for node, num_bytes, capacity in delivery.wireless_hops:
            self.vanet.record_wireless_delivery(node, t, num_bytes, capacity)
        self.aodv.ledger.host_handoff(delivery, delivery.delivered)

    def _vanet_finish_round(self, t: int, t0: float, members: Dict[str, List[str]],
                            logs, metrics: Dict, bs_result: Dict, round_wall_s: float,
                            vbytes: Dict[str, List[float]], vcomm: Dict[str, List[float]],
                            vtrain: Dict[str, Dict], va2r: Dict[str, float],
                            vpos: Dict[str, tuple]) -> None:
        cfg = self.cfg
        pid2vid = {v.pseudonym.pseudonym_id: vid for vid, v in self.vehicles.items()
                   if v.pseudonym is not None}
        exec_ms = round_wall_s * 1000.0

        def _flush_bytes_comm(node: str) -> None:
            tx, rx = vbytes.get(node, (0.0, 0.0))
            ctx, crx = vcomm.get(node, (0.0, 0.0))
            self.vanet.add_value(node, t, "bytes_tx", tx)
            self.vanet.add_value(node, t, "bytes_rx", rx)
            self.vanet.add_value(node, t, "communication_tx_ms", ctx)
            self.vanet.add_value(node, t, "communication_rx_ms", crx)

        # -- trained vehicles ------------------------------------------------
        for vid, info in vtrain.items():
            vehicle = self.vehicles[vid]
            self.vanet_quality[(vid, t)] = self._vanet_vehicle_quality(
                vehicle, info["stats"])
            self.vanet.record_duration(vid, t, "training", info["wall_s"])
            if vid in self._provision_s:
                self.vanet.record_duration(vid, t, "key_generation",
                                           self._provision_s[vid])
            ctx = vehicle.round_context
            for key in ("beacon_crypto_ms", "upload_crypto_ms"):
                crypto = ctx.get(key, {})
                if crypto.get("sign_ms"):
                    self.vanet.add_value(vid, t, "signature_generation_ms",
                                         crypto["sign_ms"])
                if crypto.get("encrypt_ms"):
                    self.vanet.add_value(vid, t, "encryption_ms", crypto["encrypt_ms"])
            ccrypto = ctx.get("config_crypto_ms", {})
            if ccrypto.get("decrypt_ms"):
                self.vanet.add_value(vid, t, "decryption_ms", ccrypto["decrypt_ms"])
            verify_ms = ccrypto.get("verify_ms", 0.0) + ctx.get("rs_verify_ms", 0.0)
            if verify_ms:
                self.vanet.add_value(vid, t, "signature_verification_ms", verify_ms)
            if vid in va2r:
                self.vanet.record_value(vid, t, "action_to_response_ms", va2r[vid])
            self.vanet.record_value(vid, t, "device_round_execution_ms", exec_ms)
            _flush_bytes_comm(vid)

        # -- batch-verification shares (pids resolved to vehicles) ------------
        for rsu in self.rsus.values():
            for receiver, pids, seconds in rsu._batch_records:
                vids = [pid2vid[p] for p in pids if p in pid2vid]
                if vids:
                    self.vanet.record_batch_duration(receiver, vids, t, seconds)
        bs_senders = [s for s in self.bs.vanet_ms.get("batch_senders", [])
                      if isinstance(s, str)]
        if bs_senders:
            self.vanet.record_batch_duration(
                SERVER_NODE, bs_senders, t,
                float(self.bs.vanet_ms.get("batch_seconds", 0.0)))

        # -- RSU rows ---------------------------------------------------------
        for name, rsu in self.rsus.items():
            served = len(members.get(name, []))
            self.vanet_quality[(name, t)] = {
                "vehicles_in_range": float(served), "vehicles_assigned": float(served)}
            ms = rsu.vanet_ms
            if ms.get("sign_ms"):
                self.vanet.add_value(name, t, "signature_generation_ms", ms["sign_ms"])
            if ms.get("encrypt_ms"):
                self.vanet.add_value(name, t, "encryption_ms", ms["encrypt_ms"])
            if ms.get("decrypt_ms"):
                self.vanet.add_value(name, t, "decryption_ms", ms["decrypt_ms"])
            self.vanet.record_value(name, t, "rsu_round_execution_ms", exec_ms)
            _flush_bytes_comm(name)

        # -- Server row --------------------------------------------------------
        clusters = bs_result.get("clusters", {}) or {}
        successful = sum(1 for n in clusters.values() if n and n > 0)
        wall = max(round_wall_s, 1e-3)
        cluster_rx = vbytes.get("BS", (0.0, 0.0))[1]
        total_served = sum(len(v) for v in members.values())
        self.vanet_quality[(SERVER_NODE, t)] = {
            "global_proxy_accuracy_pct": metrics["accuracy"] * 100.0,
            "global_proxy_f1": metrics["macro_f1"],
            "global_proxy_recall": metrics.get("macro_recall", float("nan")),
            "successful_updates": float(successful),
            "throughput_updates_per_sec": successful / wall,
            "throughput_bytes_per_sec": cluster_rx / wall,
            "model_payload_bytes_rx": sum(
                float(self.rsus[n].vanet_ms.get("cluster_app_bytes", 0.0))
                for n in clusters),
            "vehicles_in_range_total": float(total_served),
            "vehicles_assigned_total": float(total_served),
        }
        bs_ms = self.bs.vanet_ms
        if bs_ms.get("sign_ms"):
            self.vanet.add_value(SERVER_NODE, t, "signature_generation_ms",
                                 bs_ms["sign_ms"])
        if bs_ms.get("decrypt_ms"):
            self.vanet.add_value(SERVER_NODE, t, "decryption_ms", bs_ms["decrypt_ms"])
        bs_wire, _ = self.bs._broadcast_sizes
        if bs_wire:
            self.vanet.add_value(SERVER_NODE, t, "bytes_tx", bs_wire)
        self.vanet.record_value(SERVER_NODE, t, "server_round_execution_ms", exec_ms)
        _flush_bytes_comm(SERVER_NODE)

        # -- AODV overlay -------------------------------------------------------
        self._vanet_current_snapshot = self._vanet_snapshot(vpos)
        arrival = (t - 1) * cfg.round_duration_s
        serving = {vid: name for name, vids in members.items() for vid in vids}

        def _nearest(vid: str) -> Optional[str]:
            best, best_d = None, float("inf")
            for name in self.rsus:
                cx, cy = self.world.rsus[name]
                d = ((vpos[vid][0] - cx) ** 2 + (vpos[vid][1] - cy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = name, d
            return best

        for vid, vehicle in self.vehicles.items():
            ctx = vehicle.round_context
            dest = serving.get(vid) or _nearest(vid)
            if dest is None:
                continue
            if "beacon_sizes" in ctx:
                w, a = ctx["beacon_sizes"]
                self._vanet_submit(t, arrival, vid, dest, w, a)
            if "upload_sizes" in ctx:
                w, a = ctx["upload_sizes"]
                self._vanet_submit(t, arrival, vid, dest, w, a)
            if vid in serving:
                rsu = self.rsus[serving[vid]]
                w = rsu.vanet_ms.get("start_wire_bytes", 0.0)
                a = rsu.vanet_ms.get("start_app_bytes", 0.0)
                self._vanet_submit(t, arrival, serving[vid], vid, w, a)
                pid = vehicle.pseudonym.pseudonym_id if vehicle.pseudonym else None
                if pid and pid in rsu._config_sizes:
                    w, a = rsu._config_sizes[pid]
                    self._vanet_submit(t, arrival, serving[vid], vid, w, a)

    def run(self) -> Dict:
        for t in range(1, self.cfg.rounds + 1):
            self.run_round(t)
        return self.summary()

    def summary(self) -> Dict:
        ref = self.chain.reference()
        rows = self.round_rows
        return {
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in self.cfg.to_dict().items()},
            "initial_test_accuracy": self.initial_metrics["accuracy"],
            "final_test_accuracy": rows[-1]["test_accuracy"] if rows else None,
            "final_test_macro_f1": rows[-1]["test_macro_f1"] if rows else None,
            "best_test_accuracy": max(r["test_accuracy"] for r in rows) if rows else None,
            "total_selected": sum(r["selected"] for r in rows),
            "total_accepted": sum(r["accepted"] for r in rows),
            "total_dropouts": sum(r["dropouts"] for r in rows),
            "overall_dropout_rate": (sum(r["dropouts"] for r in rows)
                                     / max(sum(r["selected"] for r in rows), 1)),
            "total_upload_bytes": sum(r["upload_bytes"] for r in rows),
            "total_anomalies_rejected": sum(r["anomalies"] for r in rows),
            "malicious_accepted": sum(r["malicious_accepted"] for r in rows),
            "auth_rejected": sum(r["auth_rejected"] for r in rows),
            "adversary_attempts": dict(self.adversary.attempts),
            "revocations": [{"round": r, "identity": i} for r, i in self.revocations],
            "malicious_identities": sorted(self.malicious_ids),
            "chain_height": ref.height,
            "chain_valid": ref.verify_chain(),
            "replicas_consistent": self.chain.replicas_consistent(),
            "chain_stats": dict(self.chain.stats),
            "xi_final": {name: [float(x) for x in rsu.controller.score.xi]
                         for name, rsu in self.rsus.items()},
            "network": dict(self.network.counters),
        }

    def save(self, output_dir: Path) -> Dict[str, Path]:
        import pandas as pd
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {"rounds": output_dir / "rounds.csv", "rsu_rounds": output_dir / "rsu_rounds.csv",
                 "summary": output_dir / "summary.json"}
        pd.DataFrame(self.round_rows).to_csv(paths["rounds"], index=False)
        pd.DataFrame(self.rsu_rows).to_csv(paths["rsu_rounds"], index=False)
        paths["summary"].write_text(json.dumps(self.summary(), indent=2, default=str))
        # VANET-style artifacts (ProxyFL v1 compatible).
        paths["vanet_metrics"] = output_dir / "vanet_metrics.csv"
        self.vanet.export_csv(paths["vanet_metrics"], self.vanet_quality)
        self.aodv.ledger.export(output_dir / "vanet", self.aodv.metadata(
            traffic="BC-PAFL FL envelopes (beacon/upload uplink, broadcast/config downlink)",
            radio_configuration={
                "model": "bcpafl.channel 802.11p-style link budget",
                "bandwidth_hz": self.link.bandwidth_hz,
                "tx_power_dbm": self.link.tx_power_dbm,
                "antenna_gain_db": self.link.antenna_gain_db,
                "path_loss_1m_db": self.link.path_loss_1m_db,
                "path_loss_exponent": self.link.path_loss_exponent,
                "noise_figure_db": self.link.noise_figure_db,
                "max_rate_bps": self.link.max_rate_bps,
                "max_retries": self.link.max_retries,
                "v2v_range_m": V2V_RANGE_M,
                "v2rsu_range_m": self.cfg.rsu_range_m,
            },
            round_arrival_interval_s=self.cfg.round_duration_s))
        paths["vanet_routing_rounds"] = output_dir / "vanet_routing_rounds.csv"
        paths["vanet_routing_metadata"] = output_dir / "vanet_routing_metadata.json"
        return paths
