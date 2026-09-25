import math

import numpy as np
import pytest

from bcpafl.pomdp.belief import (ACTION_IDLE, ACTION_SELECTED, COMPONENTS, BeliefModel,
                                 JointBelief)
from bcpafl.pomdp.policy import Action, QPolicy, RewardTerms, dropout_cost
from bcpafl.pomdp.scoring import Candidate, ScoreModel

NOISE = {"trust": 0.1, "utility": 0.12, "uncertainty": 0.1, "rho": 0.5}


def _model():
    return BeliefModel(3, (10, 30, 90, 270, 810), NOISE, 0.55, 8.0)


def test_factorised_filter_equals_bruteforce_joint_eq7():
    """Eq. (7): beta'(S') = nu Z(O'|S') sum_S T(S'|S) beta(S) on the full joint space."""
    model = _model()
    rng = np.random.default_rng(0)
    for f in model.factors.values():          # non-trivial learned transitions
        f.counts[ACTION_SELECTED] += rng.random(f.counts[ACTION_SELECTED].shape) * 5
    belief = model.initial_belief(0.7)
    belief.factors["rho"] = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    belief.last_action = ACTION_SELECTED
    obs = {"trust": 0.8, "utility": 0.3, "uncertainty": 0.6, "rho": 50.0}

    T = np.array([[1.0]])
    Z = np.array([1.0])
    for name in COMPONENTS:
        f = model.factors[name]
        T = np.kron(T, f.transition(ACTION_SELECTED))
        o = math.log(obs[name]) if name == "rho" else obs[name]
        Z = np.kron(Z, f.likelihood(o))
    joint_prior = belief.as_joint()
    brute = Z * (joint_prior @ T)
    brute /= brute.sum()

    updated = model.update(belief, obs, learn=False)
    assert np.allclose(updated.as_joint(), brute, atol=1e-10)


def test_transition_learning_keeps_stochastic_rows():
    model = _model()
    belief = model.initial_belief()
    for value in (0.9, 0.85, 0.2, 0.9):
        belief = model.update(belief, {"trust": value, "utility": 0.5, "uncertainty": 0.5,
                                       "rho": 60.0})
    for f in model.factors.values():
        for a in (ACTION_IDLE, ACTION_SELECTED):
            assert np.allclose(f.transition(a).sum(axis=1), 1.0)


def test_availability_eq8_is_monotone_and_follows_belief():
    model = _model()
    b = model.initial_belief()
    qs = [model.availability(b, g) for g in (5, 20, 60, 200, 1000)]
    assert all(x >= y for x, y in zip(qs, qs[1:])) and 0 <= qs[-1] < qs[0] <= 1
    short = model.update(b, {"trust": .5, "utility": .5, "uncertainty": .5, "rho": 12}, False)
    long = model.update(b, {"trust": .5, "utility": .5, "uncertainty": .5, "rho": 700}, False)
    assert model.availability(long, 60) > model.availability(short, 60)


def test_belief_wire_round_trip_and_validation():
    b = _model().initial_belief(0.4)
    again = JointBelief.from_wire(b.to_wire())
    assert np.allclose(again.as_joint(), b.as_joint())
    bad = b.to_wire()
    bad["factors"]["trust"] = [0.9, 0.9, 0.9]
    with pytest.raises(ValueError):
        JointBelief.from_wire(bad)


def _cand(pid, q, tau, u, mu, comm=10.0, comp=10.0):
    return Candidate(pid, q, tau, u, mu, comm, comp, 100)


def test_score_eq10_and_selection_eq11_with_budgets():
    sm = ScoreModel((1.0, 2.0, 0.5, 3.0), 0.1, 4.0)
    c = _cand("a", 0.8, 0.9, 0.5, 0.2)
    expected = 0.8 ** 1.0 * 0.9 ** 2.0 * 0.5 ** 0.5 * math.exp(-3.0 * 0.2)
    assert sm.score(c) == pytest.approx(expected)
    cands = [_cand("hi", .9, .9, .9, .1), _cand("mid", .6, .6, .6, .4),
             _cand("lo", .1, .2, .1, .9)]
    chosen = [c.pseudonym_id for c in sm.select(cands, 0.05, 1e9, 1e9)]
    assert chosen == ["hi", "mid"]
    chosen = [c.pseudonym_id for c in sm.select(cands, 0.05, 15.0, 1e9)]     # Eq. (17)
    assert chosen == ["hi"]
    # Nobody clears the threshold -> best feasible candidate is still taken.
    assert [c.pseudonym_id for c in sm.select(cands, 10.0, 1e9, 1e9, 1)] == ["hi"]


def test_xi_learning_is_nonnegative_and_rewards_predictive_factor():
    sm = ScoreModel((1.0, 1.0, 1.0, 1.0), 0.5, 4.0)
    cands = [_cand(str(i), q, 0.5, 0.5, 0.5) for i, q in enumerate((0.95, 0.9, 0.2, 0.15))]
    outcomes = {"0": 1.0, "1": 1.0, "2": -1.0, "3": -1.0}   # low availability -> dropout
    for _ in range(10):
        sm.learn(outcomes, cands)
    assert sm.xi[0] > 1.0 and np.all(sm.xi >= 0)


def test_reward_eq14_and_dropout_cost_eq15():
    assert dropout_cost([0.9, 0.5, 1.0]) == pytest.approx(0.6)
    terms = RewardTerms(0.02, 1.0, 0.5, 0.3, 0.6)
    assert terms.total((20, 0.3, 0.3, 0.3, 1.0)) == pytest.approx(
        20 * 0.02 - 0.3 * 1.0 - 0.3 * 0.5 - 0.3 * 0.3 - 0.6)


def test_q_learning_prefers_rewarded_action():
    rng = np.random.default_rng(0)
    pol = QPolicy((0.1, 0.2), ("none", "q8"), ("score",), (1, 2), discount=0.5,
                  learning_rate=0.2, epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0, rng=rng)
    phi = np.ones(9)
    good, bad = Action(0.2, "q8", "score", 1), Action(0.1, "none", "score", 2)
    for _ in range(60):
        pol.record(phi, good, 1.0)
        pol.bootstrap(None)
        pol.record(phi, bad, -1.0)
        pol.bootstrap(None)
    assert pol.greedy(phi) == good
    assert len(pol.actions) == 8
