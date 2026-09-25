"""Certificateless public-key cryptography over MIRACL Core NIST P-256.

Reused from ProxyFL v1 (``crypto_protocol.py``) and re-shaped for BC-PAFL:
keys are now bound to *pseudonyms* issued by the TA rather than to fixed node
names, so one vehicle can hold many unlinkable key pairs over its lifetime.

Scheme (unchanged from v1):
    KGC master secret s, P_pub = s*P
    pseudonym AID = (AID1 = k*P, AID2 = ID xor H0(t*AID1 || T_pub))   (issued by the TA)
    partial private key  w = u + H1(AID, U, P_pub)*s,  U = u*P       (KGC)
    secret value x, X = x*P                                           (holder)
    Q = U + H2(AID, X)*X ;  pk = (Q, U, X)
    sign:   R = r*P, gamma = H3(AID, m, Q, U, R), eta = r + gamma*(w + beta*x)
    verify: eta*P == R + gamma*(Q + alpha*P_pub)
    batch:  sum(y_i eta_i)*P == sum y_i R_i + sum y_i gamma_i Q_i + (sum y_i gamma_i alpha_i) P_pub
    ECDH:   psi = x_i X_j + w_i (U_j + alpha_j P_pub)  (symmetric)
"""

from __future__ import annotations

import json
import secrets
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MIRACL_ROOT = str(Path(__file__).resolve().parent / "miracl_python")
_added = _MIRACL_ROOT not in sys.path
if _added:
    sys.path.insert(0, _MIRACL_ROOT)
try:
    from nist256 import curve as _curve  # noqa: E402
    from nist256.ecp import ECp, generator  # noqa: E402
finally:
    if _added:
        sys.path.remove(_MIRACL_ROOT)

P: ECp = generator()
q: int = int(_curve.r)
POINT_BYTES = int(_curve.EFS)


class SecurityError(ValueError):
    """Raised when key material, a signature or an envelope is invalid."""


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def random_scalar() -> int:
    """Uniform element of Z_q^* from the OS CSPRNG."""
    return secrets.randbelow(q - 1) + 1


def sha256_digest(data: bytes) -> bytes:
    return sha256(data).digest()


def point_to_bytes(point: ECp) -> bytes:
    if point.isinf():
        raise SecurityError("cannot encode point at infinity")
    x, y = point.get()
    return int(x).to_bytes(POINT_BYTES, "big") + int(y).to_bytes(POINT_BYTES, "big")


def point_from_bytes(data: bytes) -> ECp:
    if not isinstance(data, (bytes, bytearray)) or len(data) != 2 * POINT_BYTES:
        raise SecurityError("invalid P-256 point encoding")
    point = ECp()
    if not point.setxy(int.from_bytes(data[:POINT_BYTES], "big"),
                       int.from_bytes(data[POINT_BYTES:], "big")):
        raise SecurityError("point is not on P-256")
    return point


def points_equal(a: ECp, b: ECp) -> bool:
    return a.get() == b.get()


def point_add(a: ECp, b: ECp) -> ECp:
    result = a.copy()
    result.add(b)
    return result


def _hash_part(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bool):
        raw = b"\x01" if value else b"\x00"
    elif isinstance(value, int):
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    elif isinstance(value, ECp):
        raw = point_to_bytes(value)
    elif isinstance(value, tuple):
        raw = b"".join(_hash_part(item) for item in value)
    else:
        raise TypeError(f"cannot hash protocol value of type {type(value)!r}")
    return len(raw).to_bytes(4, "big") + raw


def hash_to_scalar(domain: bytes, *args: Any) -> int:
    """Domain-separated SHA-256 hash into [1, q-1] (H0..H3)."""
    digest = sha256_digest(domain + b"".join(_hash_part(a) for a in args))
    return (int.from_bytes(digest, "big") % (q - 1)) + 1


def h0_mask(t_times_aid1: ECp, t_pub: ECp) -> bytes:
    return sha256_digest(b"H0" + point_to_bytes(t_times_aid1) + point_to_bytes(t_pub))


# ----------------------------------------------------------------------
# Pseudonymous identity and public keys
# ----------------------------------------------------------------------
Aid = Tuple[ECp, bytes]


def aid_to_bytes(aid: Aid) -> bytes:
    return point_to_bytes(aid[0]) + aid[1]


def aid_from_bytes(data: bytes) -> Aid:
    if not isinstance(data, (bytes, bytearray)) or len(data) != 2 * POINT_BYTES + 32:
        raise SecurityError("invalid pseudo-identity encoding")
    return point_from_bytes(bytes(data[:2 * POINT_BYTES])), bytes(data[2 * POINT_BYTES:])


@dataclass(frozen=True)
class PublicKey:
    """Public material a verifier needs: AID plus (Q, U, X)."""

    aid: Aid
    Q: ECp
    U: ECp
    X: ECp

    def to_bytes(self) -> bytes:
        return (aid_to_bytes(self.aid) + point_to_bytes(self.Q)
                + point_to_bytes(self.U) + point_to_bytes(self.X))

    @classmethod
    def from_bytes(cls, data: bytes) -> "PublicKey":
        n = 2 * POINT_BYTES
        if not isinstance(data, (bytes, bytearray)) or len(data) != (n + 32) + 3 * n:
            raise SecurityError("invalid public-key encoding")
        data = bytes(data)
        aid = aid_from_bytes(data[:n + 32])
        rest = data[n + 32:]
        return cls(aid, point_from_bytes(rest[:n]), point_from_bytes(rest[n:2 * n]),
                   point_from_bytes(rest[2 * n:]))

    def fingerprint(self) -> str:
        """Stable identifier for this key, used as the on-chain pseudonym id."""
        return sha256_digest(b"BCPAFL/pseudonym-id/v1" + self.to_bytes()).hex()[:32]

    def reconstruct_ok(self) -> bool:
        """Q' = U + H2(AID, X) X must equal the advertised Q."""
        key = self.to_bytes()
        cached = _RECONSTRUCT_CACHE.get(key)
        if cached is None:
            beta = hash_to_scalar(b"H2", self.aid, self.X)
            cached = points_equal(point_add(self.U, beta * self.X), self.Q)
            _remember(_RECONSTRUCT_CACHE, key, cached)
        return cached


# ----------------------------------------------------------------------
# Memoisation of deterministic checks.
#
# All nodes of the simulation share one process, so e.g. seven blockchain
# replicas would otherwise recompute byte-identical public-key reconstructions
# and signature checks.  Each result is a pure function of its inputs, so a
# cache changes wall-clock time only -- never an accept/reject decision.
# ----------------------------------------------------------------------
_CACHE_LIMIT = 200_000
_RECONSTRUCT_CACHE: Dict[bytes, bool] = {}
_VERIFY_CACHE: Dict[bytes, bool] = {}


def _remember(cache: Dict[bytes, bool], key: bytes, value: bool) -> None:
    if len(cache) >= _CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def clear_verification_caches() -> None:
    _RECONSTRUCT_CACHE.clear()
    _VERIFY_CACHE.clear()


class KeyGenerationCenter:
    """KGC: holds master secret s and issues partial private keys (w, U)."""

    def __init__(self) -> None:
        self._s = random_scalar()
        self.P_pub = self._s * P

    def extract_partial_private_key(self, aid: Aid) -> Tuple[int, ECp]:
        u = random_scalar()
        U = u * P
        alpha = hash_to_scalar(b"H1", aid, U, self.P_pub)
        return (u + alpha * self._s) % q, U


class KeyPair:
    """Full certificateless key pair bound to one AID."""

    def __init__(self, aid: Aid, partial_key: Tuple[int, ECp], p_pub: ECp) -> None:
        self.aid = aid
        self.w, self.U = partial_key
        self.P_pub = p_pub
        alpha = hash_to_scalar(b"H1", aid, self.U, p_pub)
        # Eq. (11) of ProxyFL v1: validate the KGC's partial key.
        if not points_equal(self.w * P, point_add(self.U, alpha * p_pub)):
            raise SecurityError("KGC partial private key failed validation")
        self.x = random_scalar()
        self.X = self.x * P
        self.beta = hash_to_scalar(b"H2", aid, self.X)
        self.Q = point_add(self.U, self.beta * self.X)
        self.public_key = PublicKey(aid, self.Q, self.U, self.X)
        self._secret_cache: Dict[bytes, bytes] = {}

    def sign(self, message: bytes) -> Tuple[int, ECp]:
        while True:
            r = random_scalar()
            R = r * P
            gamma = hash_to_scalar(b"H3", self.aid, message, self.Q, self.U, R)
            eta = (r + gamma * (self.w + self.beta * self.x)) % q
            if eta:
                return eta, R

    def shared_secret(self, peer: PublicKey) -> bytes:
        """Pairwise secret psi_ij (cached per peer key)."""
        key = peer.to_bytes()
        cached = self._secret_cache.get(key)
        if cached is not None:
            return cached
        alpha = hash_to_scalar(b"H1", peer.aid, peer.U, self.P_pub)
        kgc_component = point_add(peer.U, alpha * self.P_pub)
        secret_point = point_add(self.x * peer.X, self.w * kgc_component)
        secret = sha256_digest(b"BCPAFL/psi/v1" + point_to_bytes(secret_point))
        self._secret_cache[key] = secret
        return secret


def signature_to_bytes(signature: Tuple[int, ECp]) -> bytes:
    eta, R = signature
    return eta.to_bytes(32, "big") + point_to_bytes(R)


def signature_from_bytes(data: bytes) -> Tuple[int, ECp]:
    if not isinstance(data, (bytes, bytearray)) or len(data) != 32 + 2 * POINT_BYTES:
        raise SecurityError("invalid signature encoding")
    eta = int.from_bytes(data[:32], "big")
    if not 0 < eta < q:
        raise SecurityError("signature scalar outside the curve order")
    return eta, point_from_bytes(bytes(data[32:]))


class Verifier:
    """Single (Eq. 12 of v1) and batch (Eq. 13 of v1) signature verification."""

    def __init__(self, p_pub: ECp) -> None:
        self.P_pub = p_pub

    def _cache_key(self, message: bytes, signature: Tuple[int, ECp], pk: PublicKey) -> bytes:
        return (sha256_digest(message) + signature_to_bytes(signature) + pk.to_bytes()
                + point_to_bytes(self.P_pub))

    def verify(self, message: bytes, signature: Tuple[int, ECp], pk: PublicKey) -> bool:
        try:
            key = self._cache_key(message, signature, pk)
        except (TypeError, ValueError, SecurityError):
            return False
        cached = _VERIFY_CACHE.get(key)
        if cached is None:
            cached = self._verify_uncached(message, signature, pk)
            _remember(_VERIFY_CACHE, key, cached)
        return cached

    def _verify_uncached(self, message: bytes, signature: Tuple[int, ECp], pk: PublicKey) -> bool:
        try:
            if not pk.reconstruct_ok():
                return False
            eta, R = signature
            if not 0 < eta < q:
                return False
            alpha = hash_to_scalar(b"H1", pk.aid, pk.U, self.P_pub)
            gamma = hash_to_scalar(b"H3", pk.aid, message, pk.Q, pk.U, R)
            right = point_add(R, gamma * point_add(pk.Q, alpha * self.P_pub))
            return points_equal(eta * P, right)
        except (TypeError, ValueError, SecurityError):
            return False

    def batch_verify(self, items: Iterable[Tuple[bytes, Tuple[int, ECp], PublicKey]]) -> bool:
        sum_eta = 0
        acc: Optional[ECp] = None
        p_pub_scalar = 0
        count = 0
        try:
            for message, (eta, R), pk in items:
                if not pk.reconstruct_ok() or not 0 < eta < q:
                    return False
                y = random_scalar()
                alpha = hash_to_scalar(b"H1", pk.aid, pk.U, self.P_pub)
                gamma = hash_to_scalar(b"H3", pk.aid, message, pk.Q, pk.U, R)
                yg = (y * gamma) % q
                sum_eta = (sum_eta + y * eta) % q
                term = point_add(y * R, yg * pk.Q)
                acc = term if acc is None else point_add(acc, term)
                p_pub_scalar = (p_pub_scalar + yg * alpha) % q
                count += 1
        except (TypeError, ValueError, SecurityError):
            return False
        if count == 0:
            return True
        return points_equal(sum_eta * P, point_add(acc, p_pub_scalar * self.P_pub))

    def verify_many(self, items: Sequence[Tuple[bytes, Tuple[int, ECp], PublicKey]]) -> list[bool]:
        """Batch-verify, falling back to per-item checks so one forgery cannot
        discard honest items.  Returns one validity flag per item."""
        if not items:
            return []
        keys = []
        for m, s, pk in items:
            try:
                keys.append(self._cache_key(m, s, pk))
            except (TypeError, ValueError, SecurityError):
                keys.append(None)
        result: list = [None if k is None else _VERIFY_CACHE.get(k) for k in keys]
        pending = [i for i, r in enumerate(result) if r is None and keys[i] is not None]
        for i, k in enumerate(keys):
            if k is None:
                result[i] = False
        if len(pending) > 1 and self.batch_verify([items[i] for i in pending]):
            for i in pending:
                result[i] = True
                _remember(_VERIFY_CACHE, keys[i], True)
        else:
            for i in pending:
                result[i] = self.verify(*items[i])
        return [bool(r) for r in result]


# ----------------------------------------------------------------------
# Authenticated encryption
# ----------------------------------------------------------------------
def aad_bytes(**fields: Any) -> bytes:
    """Canonical JSON associated data binding routing metadata to the AEAD."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt(shared_secret: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes]:
    key = sha256_digest(b"BCPAFL/AES-GCM/v1" + shared_secret)
    nonce = secrets.token_bytes(12)
    return AESGCM(key).encrypt(nonce, plaintext, aad), nonce


def decrypt(shared_secret: bytes, sealed: bytes, nonce: bytes, aad: bytes) -> bytes:
    key = sha256_digest(b"BCPAFL/AES-GCM/v1" + shared_secret)
    try:
        return AESGCM(key).decrypt(nonce, sealed, aad)
    except Exception as exc:  # cryptography raises InvalidTag
        raise SecurityError("AES-GCM authentication failed") from exc
