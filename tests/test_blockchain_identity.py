import numpy as np
import pytest

from bcpafl.blockchain import (TX_MODEL_COMMIT, TX_PSEUDONYM_BATCH, TX_TRUST_FEEDBACK,
                               LedgerError, merkle_root)


def test_pseudonyms_are_registered_on_every_replica_without_real_identity(infra):
    wallet = infra.vehicle("V001", count=3)
    block = infra.chain.commit_pending(1)
    assert block is not None and len(block.endorsements) >= infra.chain.reference().quorum()
    assert infra.chain.replicas_consistent()
    for ledger in infra.chain.replicas.values():
        for p in wallet.pseudonyms:
            assert ledger.pseudonym_status(p.pseudonym_id, 1) == (True, "valid")
            record = ledger.pseudonyms[p.pseudonym_id]
            assert "V001" not in str({k: v for k, v in record.items() if k != "public_key_obj"})
    # Pseudonyms of one vehicle are unlinkable key pairs.
    assert len({p.keypair.public_key.to_bytes() for p in wallet.pseudonyms}) == 3


def test_freshness_expiry_and_conditional_traceability(infra):
    wallet = infra.vehicle("V002", round_num=2, count=1, lifetime=2)
    infra.chain.commit_pending(2)
    ledger = infra.chain.reference()
    pid = wallet.pseudonyms[0].pseudonym_id
    assert ledger.pseudonym_status(pid, 1) == (False, "not_yet_valid")
    assert ledger.pseudonym_status(pid, 3) == (True, "valid")
    assert ledger.pseudonym_status(pid, 4) == (False, "expired")
    assert ledger.pseudonym_status("f" * 32, 2) == (False, "unregistered")
    assert infra.ta.trace(ledger.pseudonym_public_key(pid)) == "V002"


def test_revocation_and_refused_reissue(infra):
    wallet = infra.vehicle("V003", count=2)
    infra.chain.commit_pending(1)
    revoked = infra.ta.revoke("V003", 1, "test")
    infra.chain.commit_pending(1)
    ledger = infra.chain.reference()
    assert set(revoked) == {p.pseudonym_id for p in wallet.pseudonyms}
    assert all(ledger.pseudonym_status(pid, 1) == (False, "revoked") for pid in revoked)
    assert infra.ta.provision(wallet, 2, 2, 3) == []


def test_feedback_drives_tracing_and_revocation(infra):
    wallet = infra.vehicle("V004", count=1)
    infra.chain.commit_pending(1)
    pid = wallet.pseudonyms[0].pseudonym_id
    for round_num in (1, 2, 3):
        infra.chain.submit("RSU_0", TX_TRUST_FEEDBACK, {"round": round_num, "entries": [
            {"pseudonym_id": pid, "positive": 0.0, "negative": 3.0, "anomalies": 1}]})
        infra.chain.commit_pending(round_num)
        revoked = infra.ta.process_feedback(infra.chain.reference(), round_num, 0.9, 3)
    assert revoked == ["V004"]
    infra.chain.commit_pending(4)
    assert infra.chain.reference().pseudonym_status(pid, 2) == (False, "revoked")
    alpha, beta = infra.ta.reputation("V004")
    assert beta > alpha


def test_role_authorisation(infra):
    infra.chain.submit("RSU_0", TX_PSEUDONYM_BATCH, {"records": []})
    with pytest.raises(LedgerError):
        infra.chain.commit_pending(1)
    infra.chain.mempool.clear()
    infra.chain.submit("RSU_0", TX_MODEL_COMMIT, {"round": 1, "model_sha256": "00"})
    with pytest.raises(LedgerError):
        infra.chain.commit_pending(1)


def test_tampering_is_detected(infra):
    infra.vehicle("V005", count=2)
    infra.chain.commit_pending(1)
    infra.chain.submit("BS", TX_MODEL_COMMIT, {"round": 1, "model_sha256": "ab" * 32,
                                                "participants": 3})
    infra.chain.commit_pending(1)
    ledger = infra.chain.ledger("RSU_1")
    assert ledger.verify_chain()
    ledger.blocks[1].transactions[0].payload["participants"] = 999
    assert not ledger.verify_chain()
    # Other replicas are unaffected.
    assert infra.chain.ledger("RSU_0").verify_chain()


def test_merkle_root_changes_with_any_transaction():
    a = merkle_root(["00" * 32, "11" * 32, "22" * 32])
    b = merkle_root(["00" * 32, "11" * 32, "23" * 32])
    assert a != b and len(a) == 64


def test_wallet_selects_only_valid_pseudonyms(infra):
    wallet = infra.vehicle("V006", round_num=1, count=2, lifetime=1)
    infra.ta.provision(wallet, 2, 2, 2)
    rng = np.random.default_rng(0)
    picks = {wallet.select(2, rng).pseudonym_id for _ in range(30)}
    valid = {p.pseudonym_id for p in wallet.valid(2)}
    assert picks == valid and len(valid) == 2
