"""POMDP action space (Eq. 9), reward (Eqs. 14-15) and policy learning (Eq. 16).

Action  L = [v, Omega, D, f]:
    v      -- vehicle selection, induced by the policy's threshold Psi_Th (Eq. 11)
    Omega  -- update compression in {none, q8, q4, topk}
    D      -- RSU aggregation strategy in {score (Eq. 13), data (Eq. 2 weights), uniform}
    f      -- update frequency: local epochs performed before uploading

Policy  pi* = argmax E[ sum_t gamma^(t-1) R^(t) ]  (Eq. 16) is learned with
semi-gradient Q-learning over a feature summary of the belief state.  The
Q-function is additive over action dimensions, Q(b, a) = sum_d w_{d,a_d} . phi(b),
which keeps learning sample-efficient over the 72-action product space.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    threshold: float
    compression: str
    aggregation: str
    local_epochs: int

    def as_dict(self) -> Dict[str, object]:
        return {"psi_threshold": self.threshold, "compression": self.compression,
                "aggregation": self.aggregation, "local_epochs": self.local_epochs}


@dataclass
class RewardTerms:
    delta_accuracy: float
    comm_cost: float
    comp_cost: float
    divergence: float
    dropout_cost: float

    def total(self, weights: Sequence[float]) -> float:
        """Eq. (14)."""
        s1, s2, s3, s4, s5 = weights
        return (s1 * self.delta_accuracy - s2 * self.comm_cost - s3 * self.comp_cost
                - s4 * self.divergence - s5 * self.dropout_cost)


def dropout_cost(selected_q: Sequence[float]) -> float:
    """Eq. (15): C_chi = sum over selected vehicles of (1 - q_avail)."""
    return float(sum(1.0 - q for q in selected_q))


FEATURE_NAMES = ("bias", "mean_q", "mean_trust", "mean_utility", "mean_uncertainty",
                 "mean_psi", "candidates", "budget_pressure", "progress")


class QPolicy:
    def __init__(self, thresholds: Sequence[float], compression: Sequence[str],
                 aggregation: Sequence[str], epochs: Sequence[int], *, discount: float,
                 learning_rate: float, epsilon: float, epsilon_decay: float,
                 epsilon_min: float, rng: np.random.Generator) -> None:
        self.dims: Tuple[Tuple, ...] = (tuple(thresholds), tuple(compression),
                                        tuple(aggregation), tuple(epochs))
        n_feat = len(FEATURE_NAMES)
        self.weights = [np.zeros((len(options), n_feat)) for options in self.dims]
        self.gamma = float(discount)
        self.lr = float(learning_rate)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.rng = rng
        self._pending: Optional[Tuple[np.ndarray, Tuple[int, ...], float]] = None
        self.td_errors: List[float] = []

    @property
    def actions(self) -> List[Action]:
        return [Action(*combo) for combo in product(*self.dims)]

    def _indices(self, action: Action) -> Tuple[int, ...]:
        return (self.dims[0].index(action.threshold), self.dims[1].index(action.compression),
                self.dims[2].index(action.aggregation), self.dims[3].index(action.local_epochs))

    def q_value(self, phi: np.ndarray, action: Action) -> float:
        return float(sum(self.weights[d][i] @ phi for d, i in enumerate(self._indices(action))))

    def greedy(self, phi: np.ndarray) -> Action:
        choice = [self.dims[d][int(np.argmax(self.weights[d] @ phi))] for d in range(4)]
        return Action(*choice)

    def act(self, phi: np.ndarray, explore: bool = True) -> Action:
        if explore and self.rng.random() < self.epsilon:
            return Action(*[opts[int(self.rng.integers(len(opts)))] for opts in self.dims])
        return self.greedy(phi)

    def max_q(self, phi: np.ndarray) -> float:
        return float(sum(np.max(w @ phi) for w in self.weights))

    def record(self, phi: np.ndarray, action: Action, reward: float) -> None:
        """Store (phi, a, R); the TD update runs when the next belief arrives."""
        self._pending = (phi.copy(), self._indices(action), float(reward))

    def bootstrap(self, next_phi: Optional[np.ndarray]) -> Optional[float]:
        """Q-learning update for the stored transition (terminal if next_phi is None)."""
        if self._pending is None:
            return None
        phi, idx, reward = self._pending
        self._pending = None
        target = reward + (self.gamma * self.max_q(next_phi) if next_phi is not None else 0.0)
        current = float(sum(self.weights[d][i] @ phi for d, i in enumerate(idx)))
        td = target - current
        step = self.lr * td / len(idx)
        for d, i in enumerate(idx):
            self.weights[d][i] += step * phi
        self.td_errors.append(td)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td


def belief_features(expectations: Sequence[Dict[str, float]], q_values: Sequence[float],
                    psi_values: Sequence[float], budget_pressure: float,
                    progress: float, max_candidates: int) -> np.ndarray:
    """phi(b): summary statistics of the RSU's belief over its candidates."""
    if not expectations:
        return np.array([1.0, 0, 0, 0, 0, 0, 0, budget_pressure, progress])
    return np.array([
        1.0,
        float(np.mean(q_values)),
        float(np.mean([e["trust"] for e in expectations])),
        float(np.mean([e["utility"] for e in expectations])),
        float(np.mean([e["uncertainty"] for e in expectations])),
        float(np.mean(psi_values)),
        len(expectations) / max(max_candidates, 1),
        float(budget_pressure),
        float(progress),
    ])
