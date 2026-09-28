import json

import numpy as np
import pytest
import torch

from bcpafl.base_station import BaseStation
from bcpafl.compression import compress
from bcpafl.config import SimulationConfig
from bcpafl.data import NUM_CLASSES, NUM_FEATURES, to_dataset
from bcpafl.rsu import MSG_CLUSTER_UPDATE
from bcpafl.secure import seal
from bcpafl.simulation import Simulation
from bcpafl.trust import AnomalyScreen, beta_trust, ledger_trust
from bcpafl.wire_codec import encode_message


def _tiny_dataset(n=50):
    rng = np.random.default_rng(0)
    return to_dataset(rng.normal(size=(n, NUM_FEATURES)).astype(np.float32),
                      rng.integers(0, NUM_CLASSES, size=n))


def test_base_station_eq19_weights_by_participants(infra):
    bs = BaseStation(infra.bs_kp, infra.chain, _tiny_dataset(), None,
                     torch.ones(NUM_CLASSES), seed=0)
    base = bs.vector.copy()
    m0, m1 = base + 1.0, base + 4.0
    envs = []
    for name, kp, vec, n in (("RSU_0", infra.rsu_kp, m0, 3), ("RSU_1", infra.rsu1_kp, m1, 1)):
        body = encode_message({"round": 1, "n_accepted": n, "model": compress(vec, "none"),
                               "base_version": 0})
        envs.append(seal(kp, name, "BS", infra.bs_kp.public_key, MSG_CLUSTER_UPDATE, 1, body))
    result = bs.aggregate(1, envs)
    # M = (3 * M_0 + 1 * M_1) / 4  ->  base + 1.75
    assert np.allclose(bs.vector, base + 1.75, atol=1e-5)
    assert result["participants"] == 4 and bs.version == 1
    infra.chain.commit_pending(1)
    assert infra.chain.reference().model_commits[1]["model_sha256"] == result["model_sha256"]


def test_base_station_rejects_stale_or_forged_cluster_updates(infra):
    bs = BaseStation(infra.bs_kp, infra.chain, _tiny_dataset(), None,
                     torch.ones(NUM_CLASSES), seed=0)
    base = bs.vector.copy()
    stale = encode_message({"round": 1, "n_accepted": 5, "model": compress(base + 9, "none"),
                            "base_version": 7})
    # RSU_1 signs but claims to be RSU_0: key does not match the ledger record.
    spoof = encode_message({"round": 1, "n_accepted": 5, "model": compress(base + 9, "none"),
                            "base_version": 0})
    envs = [seal(infra.rsu_kp, "RSU_0", "BS", infra.bs_kp.public_key, MSG_CLUSTER_UPDATE, 1, stale),
            seal(infra.rsu1_kp, "RSU_0", "BS", infra.bs_kp.public_key, MSG_CLUSTER_UPDATE, 1,
                 spoof)]
    result = bs.aggregate(1, envs)
    assert result["rejected_cluster_updates"] == 2 and np.allclose(bs.vector, base)


def test_anomaly_screen_and_trust():
    rng = np.random.default_rng(0)
    honest = {f"h{i}": rng.normal(0.1, 0.02, size=200) for i in range(5)}
    honest["mal"] = -8.0 * honest["h0"]
    result = AnomalyScreen(3.0).screen(honest)
    assert result.rejected == ["mal"] and len(result.accepted) == 5
    assert beta_trust(0, 0) == 0.5 and beta_trust(9, 0) > beta_trust(0, 9)
    assert ledger_trust({"trust_alpha": 2, "trust_beta": 0, "positive": 1, "negative": 3}) \
        == pytest.approx(beta_trust(3, 3))


@pytest.fixture(scope="module")
def small_run():
    cfg = SimulationConfig(rounds=3, num_vehicles=12, malicious_fraction=0.2, verbose=False,
                           max_samples_per_vehicle=800, rsu_validation_rows=400,
                           forged_auth_attempts_per_round=1, revoke_malicious_after=1)
    sim = Simulation(cfg)
    summary = sim.run()
    return sim, summary


def test_end_to_end_workflow(small_run):
    sim, summary = small_run
    rows = sim.round_rows
    assert len(rows) == 3
    # Learning happened through the full RSU -> BS pipeline.
    assert summary["final_test_accuracy"] > summary["initial_test_accuracy"] + 0.2
    assert sum(r["accepted"] for r in rows) > 0
    assert all(r["bs_participants"] == r["accepted"] for r in rows)
    # Security: every adversarial attempt rejected; ledger intact and replicated.
    assert summary["auth_rejected"] >= sum(summary["adversary_attempts"].values())
    assert summary["malicious_accepted"] == 0
    assert summary["chain_valid"] and summary["replicas_consistent"]
    ledger = sim.chain.reference()
    assert set(ledger.model_commits) == {1, 2, 3}
    # Each round's committed hash is the hash of the model the BS produced.
    assert ledger.model_commits[3]["model_sha256"] == rows[-1]["model_sha256"]


def test_end_to_end_pomdp_outputs(small_run):
    sim, _ = small_run
    acted = [r for r in sim.rsu_rows if r["selected"] > 0]
    assert acted
    for r in acted:
        action = json.loads(r["action"])
        assert set(action) == {"psi_threshold", "compression", "aggregation", "local_epochs"}
        assert 0.0 <= r["mean_q_selected"] <= 1.0
        assert all(x >= 0 for x in json.loads(r["xi"]))
    assert all(r["selected"] <= r["candidates"] for r in sim.rsu_rows)


def test_every_vehicle_reports_private_metrics_each_round(small_run):
    """v1 semantics: private models train every round, selected by the POMDP or not."""
    sim, _ = small_run
    for row in sim.round_rows:
        t = row["round"]
        quality = [sim.vanet_quality.get((vid, t)) for vid in sim.vehicles]
        assert all(q is not None and np.isfinite(q["train_loss"]) for q in quality)
        assert all(0.0 <= q["private_test_accuracy_pct"] <= 100.0 for q in quality)
        # Only vehicles that trained M_i count as FL participants.
        assert sum(q["fl_participant"] for q in quality) <= row["selected"]
        server = sim.vanet_quality[("Server", t)]
        # v1 reports the global proxy on the attack benchmark.
        assert server["global_proxy_accuracy_pct"] == pytest.approx(
            row["attack_accuracy"] * 100.0)
        assert server["global_test_accuracy_pct"] == pytest.approx(
            row["test_accuracy"] * 100.0)
