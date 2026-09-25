"""RSU-side POMDP controller (Algorithm 1, lines 2-8 and 17-18).

The 7-tuple P = (S_V, O, L, T, Z, R, gamma) of Eq. (3) maps onto:
    S_V  -- :data:`belief.COMPONENTS` levels per vehicle (Eq. 4)
    O    -- :class:`Observation` built from authenticated beacons (Eq. 5)
    L    -- :class:`policy.Action` (Eq. 9)
    T, Z -- :class:`belief.FactorModel` (learned T, Gaussian Z)
    R    -- :func:`policy.RewardTerms.total` (Eq. 14)
    gamma -- ``SimulationConfig.discount`` (Eq. 16)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import channel
from ..compression import compressed_size
from ..config import SimulationConfig
from ..mobility import exit_time
from .belief import ACTION_IDLE, ACTION_SELECTED, BeliefModel, JointBelief
from .policy import Action, QPolicy, RewardTerms, belief_features, dropout_cost
from .scoring import Candidate, ScoreModel

ENVELOPE_OVERHEAD_BYTES = 700      # wire-codec + AEAD + signature + header (measured)


@dataclass
class Observation:
    """Partial observation O_Rk of one pseudonymous vehicle (Eq. 5)."""

    pseudonym_id: str
    trust: float
    utility_raw: float
    uncertainty: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    num_samples: int
    compute_rate: float           # reported samples per second
    distance_m: float


@dataclass
class Plan:
    round: int
    action: Action
    candidates: List[Candidate]
    selected: List[Candidate]
    phi: np.ndarray
    expectations: Dict[str, Dict[str, float]]
    gamma_fl: Dict[str, float]
    strategy: str


class POMDPController:
    def __init__(self, rsu_id: str, rsu_xy: Tuple[float, float], cfg: SimulationConfig,
                 num_params: int, rng: np.random.Generator) -> None:
        self.rsu_id = rsu_id
        self.rsu_xy = rsu_xy
        self.cfg = cfg
        self.num_params = num_params
        self.rng = rng
        self.link = cfg.link_params()
        self.model = BeliefModel(cfg.belief_levels, cfg.rho_level_centres, cfg.observation_noise,
                                 cfg.rho_log_spread, cfg.transition_prior_strength)
        self.beliefs: Dict[str, JointBelief] = {}
        self.score = ScoreModel(cfg.xi_init, cfg.xi_learning_rate, cfg.xi_max)
        self.policy = QPolicy(cfg.psi_thresholds, cfg.compression_options, cfg.aggregation_options,
                              cfg.local_epoch_options, discount=cfg.discount,
                              learning_rate=cfg.q_learning_rate, epsilon=cfg.exploration_start,
                              epsilon_decay=cfg.exploration_decay,
                              epsilon_min=cfg.exploration_min, rng=rng)
        self._utility_scale: Optional[float] = None
        self.max_candidates_seen = 1
        self.last_observations: Dict[str, Observation] = {}

    # ------------------------------------------------------------------
    # Observations and beliefs (Algorithm 1, lines 2-3)
    # ------------------------------------------------------------------
    def _normalise_utility(self, raws: Sequence[float]) -> None:
        if not raws:
            return
        median = float(np.median(raws))
        self._utility_scale = (median if self._utility_scale is None
                               else 0.8 * self._utility_scale + 0.2 * median)

    def utility(self, raw: float) -> float:
        scale = self._utility_scale or max(raw, 1e-6)
        return float(raw / (raw + scale))

    def observed_rho(self, obs: Observation) -> float:
        """rho~: straight-line exit time from the *reported* (noisy) kinematics."""
        return exit_time(obs.position[0], obs.position[1], obs.velocity[0], obs.velocity[1],
                         self.rsu_xy[0], self.rsu_xy[1], self.cfg.rsu_range_m)

    def export_belief(self, pid: str) -> Optional[JointBelief]:
        return self.beliefs.pop(pid, None)

    def import_belief(self, pid: str, belief: JointBelief) -> None:
        self.beliefs[pid] = belief

    def observe(self, observations: Sequence[Observation], trust_priors: Dict[str, float]) -> None:
        """Update beta(S_V) for every authenticated vehicle using Eq. (7)."""
        self._normalise_utility([o.utility_raw for o in observations])
        for obs in observations:
            belief = self.beliefs.get(obs.pseudonym_id)
            if belief is None:
                belief = self.model.initial_belief(trust_priors.get(obs.pseudonym_id))
            reading = {"trust": obs.trust, "utility": self.utility(obs.utility_raw),
                       "uncertainty": obs.uncertainty, "rho": self.observed_rho(obs)}
            self.beliefs[obs.pseudonym_id] = self.model.update(belief, reading)
            self.last_observations[obs.pseudonym_id] = obs
        self.max_candidates_seen = max(self.max_candidates_seen, len(observations))

    # ------------------------------------------------------------------
    # Availability and costs (Algorithm 1, lines 4-7)
    # ------------------------------------------------------------------
    def upload_bytes(self, compression: str) -> int:
        return compressed_size(self.num_params, compression, self.cfg.topk_fraction) + \
            ENVELOPE_OVERHEAD_BYTES

    def estimate_gamma_fl(self, obs: Observation, local_epochs: int, compression: str) -> float:
        """Gamma_FL: local computation plus the expected upload time."""
        compute = local_epochs * obs.num_samples / max(obs.compute_rate, 1e-6)
        upload = channel.expected_airtime_s(self.upload_bytes(compression), obs.distance_m,
                                            self.link)
        return compute + upload + 3 * self.cfg.per_message_overhead_s

    def link_success(self, obs: Observation, gamma_fl: float, compression: str) -> float:
        """P(upload frames delivered) at the position predicted for upload time."""
        px = obs.position[0] + obs.velocity[0] * gamma_fl
        py = obs.position[1] + obs.velocity[1] * gamma_fl
        dist = math.hypot(px - self.rsu_xy[0], py - self.rsu_xy[1])
        return channel.message_success_prob(self.upload_bytes(compression),
                                            min(dist, self.cfg.rsu_range_m), self.link)

    def availability(self, pid: str, local_epochs: int, compression: str) -> Tuple[float, float]:
        """Eq. (8) -- q_avail = P(Gamma_conn > Gamma_FL | beta) -- with the
        round deadline as a hard cap and the link-delivery probability folded in."""
        obs = self.last_observations[pid]
        gamma_fl = self.estimate_gamma_fl(obs, local_epochs, compression)
        if gamma_fl > self.cfg.round_duration_s:
            return 0.0, gamma_fl
        q = self.model.availability(self.beliefs[pid], gamma_fl)
        return q * self.link_success(obs, gamma_fl, compression), gamma_fl

    def _candidates(self, pids: Sequence[str], action_epochs: int, compression: str
                    ) -> Tuple[List[Candidate], Dict[str, Dict[str, float]], Dict[str, float]]:
        cands, exps, gammas = [], {}, {}
        for pid in pids:
            e = self.model.expectations(self.beliefs[pid])
            q, gamma_fl = self.availability(pid, action_epochs, compression)
            obs = self.last_observations[pid]
            cands.append(Candidate(pid, q, e["trust"], e["utility"], e["uncertainty"],
                                   float(self.upload_bytes(compression)),
                                   action_epochs * obs.num_samples / max(obs.compute_rate, 1e-6),
                                   obs.num_samples))
            exps[pid] = e
            gammas[pid] = gamma_fl
        for c in cands:
            c.psi = self.score.score(c)
        return cands, exps, gammas

    # ------------------------------------------------------------------
    # Planning (Algorithm 1, line 8)
    # ------------------------------------------------------------------
    def plan(self, pids: Sequence[str], round_num: int, total_rounds: int,
             strategy: str = "pomdp") -> Plan:
        cfg = self.cfg
        nominal_epochs = min(cfg.local_epoch_options)
        nominal, exps, _ = self._candidates(pids, nominal_epochs, "none")
        pressure = (len(pids) * self.upload_bytes("none")) / max(cfg.comm_budget_bytes, 1.0)
        phi = belief_features([exps[c.pseudonym_id] for c in nominal],
                              [c.q_avail for c in nominal], [c.psi for c in nominal],
                              min(pressure, 5.0), round_num / max(total_rounds, 1),
                              self.max_candidates_seen)
        # The previous round's transition can now be bootstrapped from phi.
        self.policy.bootstrap(phi)

        if strategy == "pomdp":
            action = self.policy.act(phi)
        else:
            action = Action(cfg.psi_thresholds[0], cfg.baseline_compression,
                            cfg.baseline_aggregation, cfg.baseline_local_epochs)
        cands, exps, gammas = self._candidates(pids, action.local_epochs, action.compression)

        if strategy == "pomdp":
            selected = self.score.select(cands, action.threshold, cfg.comm_budget_bytes,
                                         cfg.comp_budget_seconds, cfg.min_selected_per_rsu)
        else:
            if strategy == "random":
                order = list(self.rng.permutation(len(cands)))
                limit = max(cfg.min_selected_per_rsu,
                            int(round(cfg.random_selection_fraction * len(cands))))
                ranked = [cands[i] for i in order][:limit]
            elif strategy == "greedy":
                # Conventional: current trust and current link quality only.
                def instantaneous(c: Candidate) -> float:
                    obs = self.last_observations[c.pseudonym_id]
                    return obs.trust * channel.frame_success_prob(obs.distance_m, self.link)
                ranked = sorted(cands, key=instantaneous, reverse=True)
                ranked = ranked[:max(cfg.min_selected_per_rsu,
                                     int(round(cfg.random_selection_fraction * len(cands))))]
            else:  # "all"
                ranked = list(cands)
            selected, comm, comp = [], 0.0, 0.0
            for c in ranked:
                if comm + c.comm_cost <= cfg.comm_budget_bytes and \
                        comp + c.comp_cost <= cfg.comp_budget_seconds:
                    selected.append(c)
                    comm += c.comm_cost
                    comp += c.comp_cost
        return Plan(round_num, action, cands, selected, phi, exps, gammas, strategy)

    # ------------------------------------------------------------------
    # Learning (Algorithm 1, lines 17-18)
    # ------------------------------------------------------------------
    def finish_round(self, plan: Plan, outcomes: Dict[str, float], reward: float,
                     final_round: bool) -> None:
        selected_ids = {c.pseudonym_id for c in plan.selected}
        for c in plan.candidates:
            belief = self.beliefs.get(c.pseudonym_id)
            if belief is not None:
                belief.last_action = ACTION_SELECTED if c.pseudonym_id in selected_ids \
                    else ACTION_IDLE
        if plan.strategy != "pomdp":
            return
        self.score.learn(outcomes, plan.selected)
        self.policy.record(plan.phi, plan.action, reward)
        if final_round:
            self.policy.bootstrap(None)

    @staticmethod
    def reward_terms(delta_acc: float, bytes_sent: float, compute_s: float, divergence: float,
                     selected_q: Sequence[float], cfg: SimulationConfig) -> RewardTerms:
        return RewardTerms(delta_acc, bytes_sent / cfg.comm_budget_bytes,
                           compute_s / cfg.comp_budget_seconds, divergence,
                           dropout_cost(selected_q))
