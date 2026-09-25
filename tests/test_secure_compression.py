import numpy as np
import pytest

from bcpafl.compression import (CompressionError, ErrorFeedback, compress, compressed_size,
                                decompress)
from bcpafl.crypto.certificateless import SecurityError, Verifier
from bcpafl.network import Network
from bcpafl.channel import LinkParams
from bcpafl.secure import EnvelopeOpener, broadcast_check_input, seal, sign_broadcast


def test_seal_open_replay_recipient_and_key_checks(infra):
    wallet = infra.vehicle("V010", count=1)
    p = wallet.pseudonyms[0]
    env = seal(p.keypair, p.pseudonym_id, "RSU_0", infra.rsu_kp.public_key, "AUTH_BEACON", 1,
               b"report")
    # Through the real wire codec.
    env = Network(LinkParams(), np.random.default_rng(0)).i2i(env).message
    opener = EnvelopeOpener()
    payload, sig, pk, signed = opener.open(env, "AUTH_BEACON", p.keypair.public_key, "RSU_0",
                                           infra.rsu_kp)
    assert payload == b"report" and Verifier(infra.kgc.P_pub).verify(signed, sig, pk)
    with pytest.raises(SecurityError, match="replay"):
        opener.open(env, "AUTH_BEACON", p.keypair.public_key, "RSU_0", infra.rsu_kp)
    with pytest.raises(SecurityError):
        EnvelopeOpener().open(env, "AUTH_BEACON", p.keypair.public_key, "RSU_1", infra.rsu1_kp)
    with pytest.raises(SecurityError, match="ledger-registered"):
        EnvelopeOpener().open(env, "AUTH_BEACON", infra.bs_kp.public_key, "RSU_0", infra.rsu_kp)
    relabelled = dict(env, round=2)
    with pytest.raises(SecurityError):
        EnvelopeOpener().open(relabelled, "AUTH_BEACON", p.keypair.public_key, "RSU_0",
                              infra.rsu_kp)


def test_signed_broadcast(infra):
    msg = sign_broadcast(infra.bs_kp, "BS", "GLOBAL_MODEL", 3, b"weights")
    payload, sig, pk, signed = broadcast_check_input(msg, "GLOBAL_MODEL", infra.bs_kp.public_key)
    v = Verifier(infra.kgc.P_pub)
    assert v.verify(signed, sig, pk)
    forged = dict(msg, payload=b"poisoned")
    payload, sig, pk, signed = broadcast_check_input(forged, "GLOBAL_MODEL", infra.bs_kp.public_key)
    assert not v.verify(signed, sig, pk)


@pytest.mark.parametrize("mode,tol", [("none", 0.0), ("q8", 1 / 127), ("q4", 1 / 7)])
def test_quantisation_error_bounds(mode, tol):
    rng = np.random.default_rng(1)
    v = rng.normal(size=1000).astype(np.float32)
    payload = compress(v, mode, rng=rng)
    assert len(payload) == compressed_size(v.size, mode)
    out = decompress(payload, v.size)
    blocks = np.abs(np.pad(v, (0, 24))).reshape(-1, 256).max(axis=1)
    bound = np.repeat(blocks, 256)[:v.size] * tol + 1e-6
    assert np.all(np.abs(out - v) <= bound)


def test_topk_keeps_largest_and_sizes():
    rng = np.random.default_rng(3)
    v = (rng.permutation(np.arange(1, 101)) * rng.choice([-1, 1], 100)).astype(np.float32)
    payload = compress(v, "topk", topk_fraction=0.1)
    assert len(payload) == compressed_size(v.size, "topk", 0.1)
    out = decompress(payload, v.size)
    assert np.count_nonzero(out) == 10
    assert set(np.flatnonzero(out)) == set(np.argsort(-np.abs(v))[:10])
    assert compressed_size(4870, "q4") < compressed_size(4870, "q8") < compressed_size(4870, "none")


def test_error_feedback_carries_residual():
    rng = np.random.default_rng(2)
    ef = ErrorFeedback(500)
    total_in = np.zeros(500, dtype=np.float32)
    total_out = np.zeros(500, dtype=np.float32)
    for _ in range(20):
        d = rng.normal(size=500).astype(np.float32)
        total_in += d
        total_out += decompress(ef.compress(d, "topk", topk_fraction=0.05, rng=rng), 500)
    # With error feedback, the transmitted sum equals the true sum minus the residual.
    assert np.allclose(total_out + ef.residual, total_in, atol=1e-3)


def test_malformed_payloads_rejected():
    good = compress(np.ones(10, dtype=np.float32), "q8")
    for bad in (b"", b"XXXX" + good[4:], good[:-1]):
        with pytest.raises(CompressionError):
            decompress(bad, 10)
    with pytest.raises(CompressionError):
        decompress(good, 11)
    with pytest.raises(CompressionError):
        compress(np.array([np.nan], dtype=np.float32), "none")
