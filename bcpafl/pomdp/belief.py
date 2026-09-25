"""Belief state over the hidden vehicle state S_Vi = [tau, U, mu, rho] (Eq. 4).

Each state component takes one of a small number of discrete levels.  The
joint state space S_V is their Cartesian product.  Transitions T(S'|S, A) and
observations Z(O|S', A) factorise over components, so the Bayes filter of
Eq. (7)

    beta'(S') = nu * Z(O'|S', A) * sum_S T(S'|S, A) beta(S)

applied to a factorised prior is *exactly* the product of one filter per
component (``JointBelief.as_joint`` recovers the full joint vector; the unit
tests check the equivalence against a brute-force joint update).

Transition matrices are learned online: every update adds the expected
two-slice posterior counts xi(s, s') to Dirichlet pseudo-counts for the action
that was taken (A = vehicle selected or not in the previous round).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

COMPONENTS = ("trust", "utility", "uncertainty", "rho")
ACTION_IDLE, ACTION_SELECTED = 0, 1


def _normal_pdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


class FactorModel:
    """HMM for one state component: levels, observation model, learned T."""

    def __init__(self, name: str, centres: Sequence[float], obs_std: float,
                 prior_strength: float, persistence: float = 0.7) -> None:
        self.name = name
        self.centres = np.asarray(centres, dtype=np.float64)
        self.levels = len(self.centres)
        self.obs_std = float(obs_std)
        L = self.levels
        prior = persistence * np.eye(L) + (1 - persistence) / L
        self.counts = np.stack([prior * prior_strength, prior * prior_strength])
        self.initial = np.full(L, 1.0 / L)

    def transition(self, action: int) -> np.ndarray:
        c = self.counts[action]
        return c / c.sum(axis=1, keepdims=True)

    def likelihood(self, observation: float) -> np.ndarray:
        """Z(o | s') -- Gaussian observation noise around each level centre."""
        return _normal_pdf(np.full(self.levels, observation), self.centres, self.obs_std) + 1e-300

    def predict(self, belief: np.ndarray, action: int) -> np.ndarray:
        return belief @ self.transition(action)

    def update(self, belief: np.ndarray, action: int, observation: float,
               learn: bool = True) -> np.ndarray:
        T = self.transition(action)
        z = self.likelihood(observation)
        joint = belief[:, None] * T * z[None, :]          # xi(s, s') unnormalised
        norm = joint.sum()
        if not np.isfinite(norm) or norm <= 0:
            return self.predict(belief, action)
        joint /= norm
        if learn:
            self.counts[action] += joint
        return joint.sum(axis=0)

    def expectation(self, belief: np.ndarray) -> float:
        return float(belief @ self.centres)


@dataclass
class JointBelief:
    """beta(S_V) for one pseudonym, stored as per-component marginals."""

    factors: Dict[str, np.ndarray]
    last_action: int = ACTION_IDLE
    updates: int = 0

    def copy(self) -> "JointBelief":
        return JointBelief({k: v.copy() for k, v in self.factors.items()},
                           self.last_action, self.updates)

    def as_joint(self) -> np.ndarray:
        joint = np.array([1.0])
        for name in COMPONENTS:
            joint = np.kron(joint, self.factors[name])
        return joint

    def to_wire(self) -> Dict[str, list]:
        return {"factors": {k: v.tolist() for k, v in self.factors.items()},
                "last_action": self.last_action, "updates": self.updates}

    @classmethod
    def from_wire(cls, data: Dict) -> "JointBelief":
        factors = {k: np.asarray(v, dtype=np.float64) for k, v in data["factors"].items()}
        if set(factors) != set(COMPONENTS):
            raise ValueError("belief is missing components")
        for v in factors.values():
            if np.any(v < 0) or not np.isclose(v.sum(), 1.0, atol=1e-6):
                raise ValueError("belief marginal is not a distribution")
        return cls(factors, int(data.get("last_action", 0)), int(data.get("updates", 0)))


class BeliefModel:
    """Factor models for all four components plus availability estimation."""

    def __init__(self, levels: int, rho_centres: Sequence[float], observation_noise: Dict[str, float],
                 rho_log_spread: float, prior_strength: float) -> None:
        unit = [(i + 0.5) / levels for i in range(levels)]
        self.factors = {
            "trust": FactorModel("trust", unit, observation_noise["trust"], prior_strength),
            "utility": FactorModel("utility", unit, observation_noise["utility"], prior_strength),
            "uncertainty": FactorModel("uncertainty", unit, observation_noise["uncertainty"],
                                       prior_strength),
            # rho is modelled in the log domain (seconds are heavy-tailed).
            "rho": FactorModel("rho", np.log(np.asarray(rho_centres, dtype=np.float64)),
                               observation_noise["rho"], prior_strength),
        }
        self.rho_centres = np.asarray(rho_centres, dtype=np.float64)
        self.rho_log_spread = float(rho_log_spread)

    def initial_belief(self, trust_prior: float | None = None) -> JointBelief:
        factors = {name: f.initial.copy() for name, f in self.factors.items()}
        if trust_prior is not None:
            # Seed the trust marginal from the on-chain reputation.
            factors["trust"] = self.factors["trust"].update(
                factors["trust"], ACTION_IDLE, trust_prior, learn=False)
        return JointBelief(factors)

    def update(self, belief: JointBelief, observation: Dict[str, float],
               learn: bool = True) -> JointBelief:
        """Eq. (7) for every component, using the action taken last round."""
        new = {}
        for name in COMPONENTS:
            obs = observation[name]
            if name == "rho":
                obs = math.log(max(obs, 1.0))
            new[name] = self.factors[name].update(belief.factors[name], belief.last_action,
                                                  obs, learn=learn)
        return JointBelief(new, belief.last_action, belief.updates + 1)

    def expectations(self, belief: JointBelief) -> Dict[str, float]:
        out = {name: self.factors[name].expectation(belief.factors[name])
               for name in ("trust", "utility", "uncertainty")}
        out["rho"] = float(belief.factors["rho"] @ self.rho_centres)
        return out

    def availability(self, belief: JointBelief, gamma_fl: float) -> float:
        """Eq. (8): q_avail = P(Gamma_conn > Gamma_FL | beta(S_V)).

        Given rho level l, log(Gamma_conn) ~ N(log c_l, spread^2).
        """
        if gamma_fl <= 0:
            return 1.0
        z = (math.log(gamma_fl) - np.log(self.rho_centres)) / self.rho_log_spread
        survival = 0.5 * np.array([math.erfc(v / math.sqrt(2)) for v in z])
        return float(np.clip(belief.factors["rho"] @ survival, 0.0, 1.0))
