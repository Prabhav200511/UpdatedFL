"""Selection score Psi_i (Eq. 10), threshold selection (Eq. 11) and learnable xi.

    Psi_i = q_avail^xi1 * tau^xi2 * U^xi3 * exp(-xi4 * mu)          (Eq. 10)
    v_i   = 1  if Psi_i >= Psi_Th  else 0                            (Eq. 11)

subject to the per-round budgets

    sum_i v_i C_comm_i <= C~_comm ,   sum_i v_i C_comp_i <= C~_comp  (Eqs. 17-18)

``xi`` is learned from per-vehicle outcomes with a REINFORCE-style update on
log Psi, projected onto [0, xi_max] so every xi stays non-negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

_EPS = 1e-6


@dataclass
class Candidate:
    pseudonym_id: str
    q_avail: float
    trust: float
    utility: float
    uncertainty: float
    comm_cost: float        # bytes if selected (Eq. 17)
    comp_cost: float        # seconds of computation if selected (Eq. 18)
    num_samples: int
    psi: float = 0.0

    def log_features(self) -> np.ndarray:
        """d log Psi / d xi."""
        return np.array([math.log(max(self.q_avail, _EPS)), math.log(max(self.trust, _EPS)),
                         math.log(max(self.utility, _EPS)), -self.uncertainty])


class ScoreModel:
    def __init__(self, xi: Sequence[float], learning_rate: float, xi_max: float) -> None:
        self.xi = np.clip(np.asarray(xi, dtype=np.float64), 0.0, xi_max)
        self.lr = float(learning_rate)
        self.xi_max = float(xi_max)
        self.history: List[np.ndarray] = [self.xi.copy()]

    def score(self, c: Candidate) -> float:
        """Eq. (10)."""
        return float(math.exp(float(self.xi @ c.log_features())))

    def select(self, candidates: Sequence[Candidate], threshold: float, comm_budget: float,
               comp_budget: float, min_selected: int = 1) -> List[Candidate]:
        """Eq. (11) under the Eq. (17)-(18) constraints.

        Candidates are admitted in descending Psi order while Psi >= threshold
        and the budgets hold.  If nobody clears the threshold the best
        ``min_selected`` feasible candidates are taken so the RSU never stalls.
        """
        for c in candidates:
            c.psi = self.score(c)
        ranked = sorted(candidates, key=lambda c: c.psi, reverse=True)
        chosen: List[Candidate] = []
        comm = comp = 0.0
        for c in ranked:
            if c.psi < threshold and len(chosen) >= min_selected:
                break
            if comm + c.comm_cost > comm_budget or comp + c.comp_cost > comp_budget:
                continue
            chosen.append(c)
            comm += c.comm_cost
            comp += c.comp_cost
        return chosen

    def learn(self, outcomes: Dict[str, float], selected: Sequence[Candidate]) -> np.ndarray:
        """Policy-gradient step on xi from per-vehicle outcomes.

        ``outcomes[pid]`` is a scalar reward for a selected vehicle (quality of
        an accepted update, negative for a dropout or rejected update).  The
        centred features remove the common shift in log Psi so the update only
        re-weights *which* factor separated good from bad selections.
        """
        items = [(outcomes[c.pseudonym_id], c.log_features()) for c in selected
                 if c.pseudonym_id in outcomes]
        if len(items) < 2:
            return self.xi
        rewards = np.array([r for r, _ in items])
        feats = np.stack([f for _, f in items])
        advantage = rewards - rewards.mean()
        centred = feats - feats.mean(axis=0)
        scale = centred.std(axis=0) + 1e-3
        grad = (advantage[:, None] * centred / scale).mean(axis=0)
        self.xi = np.clip(self.xi + self.lr * grad, 0.0, self.xi_max)
        self.history.append(self.xi.copy())
        return self.xi
