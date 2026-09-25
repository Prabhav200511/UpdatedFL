import pytest

from bcpafl.crypto.certificateless import (
    P, KeyGenerationCenter, KeyPair, PublicKey, SecurityError, Verifier, clear_verification_caches,
    decrypt, encrypt, random_scalar, signature_from_bytes, signature_to_bytes,
)


def _keypair(kgc):
    aid = (random_scalar() * P, bytes(32))
    return KeyPair(aid, kgc.extract_partial_private_key(aid), kgc.P_pub)


def test_sign_verify_and_forgery_rejected():
    kgc = KeyGenerationCenter()
    kp, other = _keypair(kgc), _keypair(kgc)
    v = Verifier(kgc.P_pub)
    sig = kp.sign(b"model update")
    assert v.verify(b"model update", sig, kp.public_key)
    assert not v.verify(b"tampered", sig, kp.public_key)
    assert not v.verify(b"model update", sig, other.public_key)


def test_batch_verification_isolates_forgery():
    clear_verification_caches()
    kgc = KeyGenerationCenter()
    kps = [_keypair(kgc) for _ in range(4)]
    items = [(f"m{i}".encode(), kp.sign(f"m{i}".encode()), kp.public_key) for i, kp in enumerate(kps)]
    v = Verifier(kgc.P_pub)
    assert v.batch_verify(items)
    items[2] = (b"forged", items[2][1], items[2][2])
    assert not v.batch_verify(items)
    assert v.verify_many(items) == [True, True, False, True]


def test_partial_key_from_wrong_kgc_is_rejected():
    kgc, rogue = KeyGenerationCenter(), KeyGenerationCenter()
    aid = (random_scalar() * P, bytes(32))
    with pytest.raises(SecurityError):
        KeyPair(aid, rogue.extract_partial_private_key(aid), kgc.P_pub)


def test_shared_secret_is_symmetric_and_aead_detects_tampering():
    kgc = KeyGenerationCenter()
    a, b = _keypair(kgc), _keypair(kgc)
    assert a.shared_secret(b.public_key) == b.shared_secret(a.public_key)
    sealed, nonce = encrypt(a.shared_secret(b.public_key), b"payload", b"aad")
    assert decrypt(b.shared_secret(a.public_key), sealed, nonce, b"aad") == b"payload"
    with pytest.raises(SecurityError):
        decrypt(b.shared_secret(a.public_key), sealed, nonce, b"other aad")
    with pytest.raises(SecurityError):
        decrypt(b.shared_secret(a.public_key), sealed[:-1] + bytes([sealed[-1] ^ 1]), nonce, b"aad")


def test_encodings_round_trip():
    kgc = KeyGenerationCenter()
    kp = _keypair(kgc)
    pk = PublicKey.from_bytes(kp.public_key.to_bytes())
    assert pk.to_bytes() == kp.public_key.to_bytes() and pk.reconstruct_ok()
    sig = kp.sign(b"x")
    assert signature_to_bytes(signature_from_bytes(signature_to_bytes(sig))) == signature_to_bytes(sig)
    with pytest.raises(SecurityError):
        signature_from_bytes(b"\x00" * 96)       # eta = 0 is outside Z_q^*
