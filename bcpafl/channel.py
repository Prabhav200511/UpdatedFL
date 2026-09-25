"""IEEE 802.11p-style V2I link model with frame errors and ARQ.

Unlike ProxyFL v1's measurement-only model, the channel here *does* affect the
protocol: frames can be lost, retransmissions consume airtime, and a message
whose frame exhausts its retries is not delivered.  This is one of the ways a
selected vehicle fails to contribute, which the POMDP must anticipate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FRAME_BYTES = 1500


@dataclass(frozen=True)
class LinkParams:
    bandwidth_hz: float = 10e6
    tx_power_dbm: float = 23.0
    antenna_gain_db: float = 10.0
    path_loss_1m_db: float = 46.4
    path_loss_exponent: float = 2.7
    noise_figure_db: float = 9.0
    max_rate_bps: float = 6e6
    snr_reference: float = 1.0
    max_retries: int = 3
    wired_rate_bps: float = 1e9
    wired_latency_s: float = 0.002


def snr_linear(distance_m: float, p: LinkParams) -> float:
    d = max(float(distance_m), 1.0)
    path_loss = p.path_loss_1m_db + 10 * p.path_loss_exponent * math.log10(d)
    rx_dbm = p.tx_power_dbm + p.antenna_gain_db - path_loss
    noise_dbm = -174.0 + 10 * math.log10(p.bandwidth_hz) + p.noise_figure_db
    return 10 ** ((rx_dbm - noise_dbm) / 10)


def capacity_bps(distance_m: float, p: LinkParams) -> float:
    shannon = p.bandwidth_hz * math.log2(1 + snr_linear(distance_m, p))
    return float(min(p.max_rate_bps, max(shannon, 1e3)))


def frame_success_prob(distance_m: float, p: LinkParams) -> float:
    return float(1.0 - math.exp(-snr_linear(distance_m, p) / p.snr_reference))


def message_success_prob(num_bytes: int, distance_m: float, p: LinkParams,
                         retries: int | None = None) -> float:
    """P(all frames delivered within the retry budget)."""
    frames = max(1, math.ceil(num_bytes / FRAME_BYTES))
    attempts = (p.max_retries if retries is None else retries) + 1
    per_frame = 1.0 - (1.0 - frame_success_prob(distance_m, p)) ** attempts
    return per_frame ** frames


def expected_airtime_s(num_bytes: int, distance_m: float, p: LinkParams) -> float:
    ps = max(frame_success_prob(distance_m, p), 1e-3)
    return num_bytes * 8 / capacity_bps(distance_m, p) / ps


@dataclass
class Transmission:
    delivered: bool
    airtime_s: float
    attempts: int
    num_bytes: int


def transmit_unicast(num_bytes: int, distance_m: float, p: LinkParams,
                     rng: np.random.Generator) -> Transmission:
    """Simulate ARQ frame by frame; returns delivery outcome and airtime."""
    frames = max(1, math.ceil(num_bytes / FRAME_BYTES))
    ps = frame_success_prob(distance_m, p)
    rate = capacity_bps(distance_m, p)
    attempts = 0
    for _frame in range(frames):
        for _ in range(p.max_retries + 1):
            attempts += 1
            if rng.random() < ps:
                break
        else:
            return Transmission(False, attempts * FRAME_BYTES * 8 / rate, attempts, num_bytes)
    return Transmission(True, (num_bytes * 8 + (attempts - frames) * FRAME_BYTES * 8) / rate,
                        attempts, num_bytes)


def broadcast_reception(num_bytes: int, distance_m: float, p: LinkParams,
                        rng: np.random.Generator, repetitions: int = 2) -> bool:
    """Unacknowledged broadcast repeated ``repetitions`` times (no ARQ)."""
    return bool(rng.random() < message_success_prob(num_bytes, distance_m, p,
                                                    retries=repetitions - 1))
