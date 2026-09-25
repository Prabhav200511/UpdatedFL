"""Protected messages: sign-then-encrypt envelopes and signed broadcasts.

Changes relative to ProxyFL v1's envelope:

* The signature travels **inside** the AES-GCM ciphertext (v1 limitation L7:
  a cleartext signature over the plaintext let an observer confirm guesses).
* Recipients keep a **nonce cache** in addition to round binding (v1 L8).
* The sender's public key is looked up on the ledger by the receiver; the key
  carried in the envelope must match the ledger record byte for byte.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Set, Tuple

from .crypto.certificateless import (
    POINT_BYTES, KeyPair, PublicKey, SecurityError, aad_bytes, decrypt, encrypt,
    signature_from_bytes, signature_to_bytes,
)

SIG_BYTES = 32 + 2 * POINT_BYTES


def _aad(msg_type: str, sender: str, recipient: str, round_num: int, sender_pk: bytes) -> bytes:
    return aad_bytes(type=msg_type, sender=sender, recipient=recipient, round=round_num,
                     sender_pk=hashlib.sha256(sender_pk).hexdigest())


def _signed_body(msg_type: str, sender: str, recipient: str, round_num: int,
                 payload: bytes) -> bytes:
    return (aad_bytes(type=msg_type, sender=sender, recipient=recipient, round=round_num)
            + hashlib.sha256(payload).digest())


def seal(sender_kp: KeyPair, sender_id: str, recipient_id: str, recipient_pk: PublicKey,
         msg_type: str, round_num: int, payload: bytes,
         timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    import time as _time
    _t0 = _time.perf_counter()
    signature = sender_kp.sign(_signed_body(msg_type, sender_id, recipient_id, round_num, payload))
    _t1 = _time.perf_counter()
    sender_pk = sender_kp.public_key.to_bytes()
    ciphertext, nonce = encrypt(sender_kp.shared_secret(recipient_pk),
                                signature_to_bytes(signature) + payload,
                                _aad(msg_type, sender_id, recipient_id, round_num, sender_pk))
    _t2 = _time.perf_counter()
    if timings is not None:
        timings["sign_ms"] = timings.get("sign_ms", 0.0) + (_t1 - _t0) * 1000.0
        timings["encrypt_ms"] = timings.get("encrypt_ms", 0.0) + (_t2 - _t1) * 1000.0
    return {"type": msg_type, "sender": sender_id, "recipient": recipient_id,
            "round": round_num, "sender_pk": sender_pk, "nonce": nonce, "ciphertext": ciphertext}


class EnvelopeOpener:
    """Recipient-side decryption with replay protection.

    ``open`` returns ``(payload, signature, sender_pk, signed_message)``; the
    signature is *not* checked here so receivers can batch-verify many
    envelopes at once with :meth:`Verifier.verify_many`.
    """

    def __init__(self) -> None:
        self._seen: Set[bytes] = set()

    def open(self, envelope: Dict[str, Any], expected_type: str, ledger_pk: Optional[PublicKey],
             recipient_id: str, recipient_kp: KeyPair,
             timings: Optional[Dict[str, float]] = None
             ) -> Tuple[bytes, Tuple[int, Any], PublicKey, bytes]:
        try:
            msg_type = envelope["type"]
            sender = envelope["sender"]
            recipient = envelope["recipient"]
            round_num = envelope["round"]
            sender_pk_bytes = envelope["sender_pk"]
            nonce = envelope["nonce"]
            ciphertext = envelope["ciphertext"]
        except (KeyError, TypeError) as exc:
            raise SecurityError("malformed envelope") from exc
        if msg_type != expected_type or recipient != recipient_id:
            raise SecurityError("envelope type or recipient mismatch")
        if not isinstance(round_num, int) or round_num < 0 or not isinstance(sender, str):
            raise SecurityError("invalid envelope header")
        if ledger_pk is None or sender_pk_bytes != ledger_pk.to_bytes():
            raise SecurityError("sender key is not the ledger-registered key")
        if not isinstance(nonce, bytes) or len(nonce) != 12:
            raise SecurityError("invalid nonce")
        if nonce in self._seen:
            raise SecurityError("replayed envelope")
        import time as _time
        _t0 = _time.perf_counter()
        plaintext = decrypt(recipient_kp.shared_secret(ledger_pk), ciphertext, nonce,
                            _aad(msg_type, sender, recipient, round_num, sender_pk_bytes))
        if timings is not None:
            timings["decrypt_ms"] = timings.get("decrypt_ms", 0.0) + (
                _time.perf_counter() - _t0) * 1000.0
        self._seen.add(nonce)
        if len(plaintext) < SIG_BYTES:
            raise SecurityError("envelope too short for a signature")
        signature = signature_from_bytes(plaintext[:SIG_BYTES])
        payload = plaintext[SIG_BYTES:]
        return payload, signature, ledger_pk, _signed_body(msg_type, sender, recipient,
                                                           round_num, payload)


def sign_broadcast(sender_kp: KeyPair, sender_id: str, msg_type: str, round_num: int,
                   payload: bytes) -> Dict[str, Any]:
    """Integrity-protected one-to-many message (e.g. the global model broadcast)."""
    signature = sender_kp.sign(_signed_body(msg_type, sender_id, "*", round_num, payload))
    return {"type": msg_type, "sender": sender_id, "recipient": "*", "round": round_num,
            "payload": payload, "signature": signature_to_bytes(signature)}


def broadcast_check_input(message: Dict[str, Any], expected_type: str,
                          ledger_pk: Optional[PublicKey]) -> Tuple[bytes, Tuple[int, Any], PublicKey, bytes]:
    if message.get("type") != expected_type or message.get("recipient") != "*":
        raise SecurityError("broadcast type mismatch")
    if ledger_pk is None:
        raise SecurityError("unknown broadcast sender")
    payload = message.get("payload")
    if not isinstance(payload, bytes):
        raise SecurityError("broadcast payload must be bytes")
    signed = _signed_body(expected_type, message["sender"], "*", message["round"], payload)
    return payload, signature_from_bytes(message["signature"]), ledger_pk, signed
