"""Vehicle mobility, RSU coverage and remaining-connection-time estimation.

Vehicles move continuously in simulated time with a random-walk heading and
reflect off the area boundary, so they genuinely drive into, through and out
of RSU coverage -- the source of the dropouts the POMDP must predict.  (v1
fixed every vehicle's speed at zero, so no vehicle ever left coverage.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

CONNECTION_TIME_CAP_S = 3600.0


@dataclass
class VehicleState:
    x: float
    y: float
    speed: float
    heading: float

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)


def exit_time(px: float, py: float, vx: float, vy: float, cx: float, cy: float,
              radius: float) -> float:
    """Time until a straight-line trajectory leaves the disc (0 if outside)."""
    dx, dy = px - cx, py - cy
    c = dx * dx + dy * dy - radius * radius
    if c > 0:
        return 0.0
    a = vx * vx + vy * vy
    if a < 1e-9:
        return CONNECTION_TIME_CAP_S
    b = 2 * (dx * vx + dy * vy)
    t = (-b + math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return float(min(max(t, 0.0), CONNECTION_TIME_CAP_S))


class World:
    def __init__(self, rsus: Iterable[Tuple[str, float, float]], rsu_range: float,
                 half_width: float, speed_range: Tuple[float, float], heading_jitter: float,
                 rng: np.random.Generator) -> None:
        self.rsus: Dict[str, Tuple[float, float]] = {name: (x, y) for name, x, y in rsus}
        self.rsu_range = float(rsu_range)
        self.half_width = float(half_width)
        self.speed_range = speed_range
        self.heading_jitter = float(heading_jitter)
        self.rng = rng
        self.vehicles: Dict[str, VehicleState] = {}
        self.time_s = 0.0

    def spawn(self, vid: str, near_rsu: Optional[str] = None) -> VehicleState:
        if near_rsu is not None:
            cx, cy = self.rsus[near_rsu]
            angle = self.rng.uniform(0, 2 * math.pi)
            radius = self.rsu_range * math.sqrt(self.rng.uniform(0, 1)) * 0.95
            x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        else:
            x, y = self.rng.uniform(-self.half_width, self.half_width, size=2)
        state = VehicleState(float(x), float(y), float(self.rng.uniform(*self.speed_range)),
                             float(self.rng.uniform(0, 2 * math.pi)))
        self.vehicles[vid] = state
        return state

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        w = self.half_width
        for state in self.vehicles.values():
            state.heading += self.rng.normal(0.0, self.heading_jitter) * math.sqrt(dt)
            vx, vy = state.velocity
            x, y = state.x + vx * dt, state.y + vy * dt
            if abs(x) > w:
                x = math.copysign(2 * w - abs(x), x)
                state.heading = math.pi - state.heading
            if abs(y) > w:
                y = math.copysign(2 * w - abs(y), y)
                state.heading = -state.heading
            state.x, state.y = x, y

    def advance_to(self, t_s: float, max_step: float = 1.0) -> None:
        while self.time_s < t_s - 1e-9:
            dt = min(max_step, t_s - self.time_s)
            self._step(dt)
            self.time_s += dt

    # ------------------------------------------------------------------
    def distance(self, vid: str, rsu: str) -> float:
        s = self.vehicles[vid]
        cx, cy = self.rsus[rsu]
        return math.hypot(s.x - cx, s.y - cy)

    def in_range(self, vid: str, rsu: str) -> bool:
        return self.distance(vid, rsu) <= self.rsu_range

    def associated_rsu(self, vid: str) -> Optional[str]:
        best, best_d = None, float("inf")
        for rsu in self.rsus:
            d = self.distance(vid, rsu)
            if d <= self.rsu_range and d < best_d:
                best, best_d = rsu, d
        return best

    def members(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {rsu: [] for rsu in self.rsus}
        for vid in self.vehicles:
            rsu = self.associated_rsu(vid)
            if rsu is not None:
                result[rsu].append(vid)
        return result

    def remaining_connection_time(self, vid: str, rsu: str) -> float:
        """Ground-truth straight-line exit time from the vehicle's true state."""
        s = self.vehicles[vid]
        vx, vy = s.velocity
        cx, cy = self.rsus[rsu]
        return exit_time(s.x, s.y, vx, vy, cx, cy, self.rsu_range)
