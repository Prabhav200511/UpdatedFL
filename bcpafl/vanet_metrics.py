"""VANET-style per-node-per-round metrics (ProxyFL v1 compatible schema).

Collects the same ``(node, round)`` rows as ProxyFL v1's ``vanet_metrics.csv``
so v1's plot suite can be reproduced from a BC-PAFL run: per-vehicle training
quality, per-operation crypto timings, communication latencies, modeled
wireless goodput, coverage counts, byte counters and OBU energy estimates.

Energy model (identical constants to v1 ``config.py``):
``E = 10.88 W * x_op * t_ms / 1000`` with x = 1.0 (train), 0.4 (crypto),
0.6 (communication), 0.2 (idle).  ``energy_total_j`` covers security +
communication only, matching v1.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

OBU_PEAK_POWER_W = 10.88
X_OP_TRAIN = 1.0
X_OP_CRYPTO = 0.4
X_OP_COMM = 0.6
X_OP_IDLE = 0.2

SERVER_NODE = "Server"

VANET_COLUMNS = [
    "node", "round", "fl_participant", "train_loss", "proxy_train_loss",
    "train_accuracy_pct", "private_test_accuracy_pct",
    "epsilon", "delta", "global_proxy_accuracy_pct", "global_test_accuracy_pct",
    "global_proxy_f1",
    "global_proxy_recall", "successful_updates",
    "throughput_updates_per_sec", "throughput_bytes_per_sec",
    "vanet_wireless_bits", "vanet_airtime_s",
    "vanet_link_capacity_bps", "vanet_goodput_bps",
    "vehicles_in_range", "vehicles_assigned",
    "vehicles_in_range_total", "vehicles_assigned_total",
    "bytes_tx", "bytes_rx", "model_payload_bytes_rx", "training_ms",
    "key_generation_ms", "signature_generation_ms", "signature_verification_ms",
    "batch_verification_ms", "batch_verification_receiver_ms", "encryption_ms",
    "decryption_ms", "communication_tx_ms", "communication_rx_ms",
    "security_latency_ms", "communication_latency_ms", "action_to_response_ms",
    "end_to_end_time_ms", "device_round_execution_ms", "rsu_round_execution_ms",
    "server_round_execution_ms", "energy_training_j", "energy_security_j",
    "energy_communication_j", "energy_total_j", "energy_idle_j", "idle_latency_ms",
]


def energy_joules(duration_ms: float, utilization: float) -> float:
    return OBU_PEAK_POWER_W * utilization * max(float(duration_ms), 0.0) / 1000.0


class VanetTracker:
    """Accumulates ``(node, round)`` metric rows for one simulation."""

    def __init__(self) -> None:
        self._rows: Dict[tuple, Dict[str, float]] = defaultdict(dict)

    # -- raw recorders ----------------------------------------------------
    def record_duration(self, node: str, round_num: int, metric: str,
                        duration_seconds: float) -> None:
        row = self._rows[(node, round_num)]
        key = metric if metric.endswith("_ms") else f"{metric}_ms"
        row[key] = row.get(key, 0.0) + max(duration_seconds, 0.0) * 1000.0

    def record_value(self, node: str, round_num: int, metric: str, value: float) -> None:
        self._rows[(node, round_num)][metric] = float(value)

    def add_value(self, node: str, round_num: int, metric: str, value: float) -> None:
        row = self._rows[(node, round_num)]
        row[metric] = row.get(metric, 0.0) + float(value)

    def record_bytes(self, node: str, round_num: int, direction: str,
                     num_bytes: int) -> None:
        if direction not in {"tx", "rx"}:
            raise ValueError("direction must be 'tx' or 'rx'")
        self.add_value(node, round_num, f"bytes_{direction}", max(num_bytes, 0))

    def record_wireless_delivery(self, node: str, round_num: int, num_wire_bytes: int,
                                 capacity_bps: float) -> None:
        bits = float(max(num_wire_bytes, 0) * 8)
        if capacity_bps <= 0:
            raise ValueError("wireless capacity must be positive")
        row = self._rows[(node, round_num)]
        row["vanet_wireless_bits"] = row.get("vanet_wireless_bits", 0.0) + bits
        row["vanet_airtime_s"] = row.get("vanet_airtime_s", 0.0) + bits / capacity_bps
        row["vanet_capacity_sum_bps"] = row.get("vanet_capacity_sum_bps", 0.0) + capacity_bps
        row["vanet_capacity_samples"] = row.get("vanet_capacity_samples", 0.0) + 1.0

    def record_batch_duration(self, receiver: str, participants: Sequence[str],
                              round_num: int, duration_seconds: float) -> None:
        """Receiver wall time plus an equal share per participant (v1 semantics)."""
        participants = [p for p in participants if p]
        duration_ms = max(duration_seconds, 0.0) * 1000.0
        rec = self._rows[(receiver, round_num)]
        rec["batch_verification_receiver_ms"] = rec.get(
            "batch_verification_receiver_ms", 0.0) + duration_ms
        if participants:
            share = duration_ms / len(participants)
            for node in participants:
                row = self._rows[(node, round_num)]
                row["batch_verification_ms"] = row.get("batch_verification_ms", 0.0) + share

    # -- derived + export -------------------------------------------------
    @staticmethod
    def _derived(row: Mapping[str, float]) -> Dict[str, float]:
        training = row.get("training_ms", 0.0)
        security = (row.get("key_generation_ms", 0.0)
                    + row.get("signature_generation_ms", 0.0)
                    + row.get("signature_verification_ms", 0.0)
                    + row.get("batch_verification_ms", 0.0)
                    + row.get("encryption_ms", 0.0)
                    + row.get("decryption_ms", 0.0))
        communication = row.get("communication_tx_ms", 0.0) + row.get(
            "communication_rx_ms", 0.0)
        execution = max(row.get("device_round_execution_ms", 0.0),
                        row.get("rsu_round_execution_ms", 0.0),
                        row.get("server_round_execution_ms", 0.0))
        idle = max(execution - (training + security + communication), 0.0)
        bits = row.get("vanet_wireless_bits", 0.0)
        airtime = row.get("vanet_airtime_s", 0.0)
        samples = row.get("vanet_capacity_samples", 0.0)
        return {
            "security_latency_ms": security,
            "communication_latency_ms": communication,
            "end_to_end_time_ms": training + security + communication + idle,
            "energy_training_j": energy_joules(training, X_OP_TRAIN),
            "energy_security_j": energy_joules(security, X_OP_CRYPTO),
            "energy_communication_j": energy_joules(communication, X_OP_COMM),
            "energy_idle_j": energy_joules(idle, X_OP_IDLE),
            "energy_total_j": energy_joules(security, X_OP_CRYPTO)
            + energy_joules(communication, X_OP_COMM),
            "idle_latency_ms": idle,
            "vanet_link_capacity_bps": (row.get("vanet_capacity_sum_bps", 0.0) / samples
                                        if samples > 0 else 0.0),
            "vanet_goodput_bps": (bits / airtime if airtime > 0 else 0.0),
        }

    def rows(self, quality: Optional[Mapping[tuple, Mapping[str, float]]] = None) -> List[dict]:
        quality = quality or {}
        result = []
        for node, round_num in sorted(set(self._rows) | set(quality),
                                     key=lambda k: (k[1], k[0])):
            row: Dict[str, float] = {"node": node, "round": round_num}
            row.update(self._rows.get((node, round_num), {}))
            row.update(quality.get((node, round_num), {}))
            row.update(self._derived(row))
            result.append(row)
        return result

    def export_csv(self, path, quality: Optional[Mapping[tuple, Mapping[str, float]]] = None):
        frame = pd.DataFrame(self.rows(quality))
        for column in VANET_COLUMNS:
            if column not in frame:
                frame[column] = 0.0
        frame = frame[VANET_COLUMNS].sort_values(["round", "node"])
        frame.to_csv(path, index=False)
        return path
