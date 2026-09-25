"""Message transport for the simulated VANET.

Every message is really serialised with the wire codec and deserialised at
the receiver, so byte counts are exact and malformed frames are rejected the
same way a socket transport would reject them.  Delivery over V2I links goes
through the frame-error / ARQ model in :mod:`channel`; RSU <-> base-station
links are wired (I2I) and lossless.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import channel
from .wire_codec import decode_message, encode_message

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass
class Delivery:
    delivered: bool
    message: Optional[Dict[str, Any]]
    num_bytes: int
    airtime_s: float


class Network:
    def __init__(self, link: channel.LinkParams, rng: np.random.Generator) -> None:
        self.link = link
        self.rng = rng
        self.counters: Dict[str, float] = defaultdict(float)

    def _encode(self, message: Dict[str, Any]) -> bytes:
        data = encode_message(message)
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("message exceeds the frame cap")
        return data

    def _account(self, kind: str, msg_type: str, num_bytes: int, airtime: float,
                 delivered: bool) -> None:
        self.counters[f"{kind}_bytes"] += num_bytes
        self.counters[f"{kind}_airtime_s"] += airtime
        self.counters[f"{kind}_messages"] += 1
        self.counters[f"type_{msg_type}_bytes"] += num_bytes
        if not delivered:
            self.counters[f"{kind}_lost"] += 1

    def v2i(self, message: Dict[str, Any], distance_m: float) -> Delivery:
        """Unicast over the wireless link with ARQ."""
        data = self._encode(message)
        tx = channel.transmit_unicast(len(data), distance_m, self.link, self.rng)
        self._account("v2i", str(message.get("type")), len(data), tx.airtime_s, tx.delivered)
        return Delivery(tx.delivered, decode_message(data) if tx.delivered else None,
                        len(data), tx.airtime_s)

    def broadcast(self, message: Dict[str, Any], distances_m: Dict[str, float],
                  repetitions: int = 2) -> Dict[str, Delivery]:
        """One-to-many V2I broadcast: airtime is paid once, reception is per vehicle."""
        data = self._encode(message)
        worst = max(distances_m.values(), default=1.0)
        airtime = repetitions * len(data) * 8 / channel.capacity_bps(worst, self.link)
        self._account("broadcast", str(message.get("type")), len(data) * repetitions, airtime, True)
        result = {}
        for vid, dist in distances_m.items():
            ok = channel.broadcast_reception(len(data), dist, self.link, self.rng, repetitions)
            result[vid] = Delivery(ok, decode_message(data) if ok else None, len(data),
                                   airtime)
        return result

    def i2i(self, message: Dict[str, Any]) -> Delivery:
        """Wired RSU <-> base-station (or RSU <-> RSU) link."""
        data = self._encode(message)
        airtime = self.link.wired_latency_s + len(data) * 8 / self.link.wired_rate_bps
        self._account("i2i", str(message.get("type")), len(data), airtime, True)
        return Delivery(True, decode_message(data), len(data), airtime)
