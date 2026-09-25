"""Safe outer-wire encoding for BC-PAFL network messages (reused from ProxyFL v1)."""

import base64
import binascii
import json
import math
from collections.abc import Mapping
from typing import Any


class WireCodecError(ValueError):
    """Raised when a value cannot be safely represented on the wire."""


def _to_wire(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise WireCodecError("non-finite numbers are not allowed")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "__bcpafl_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _to_wire(item) for key, item in value.items()}
    raise WireCodecError(f"unsupported wire value: {type(value).__name__}")


def _from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    if isinstance(value, dict):
        if value.get("__bcpafl_type__") == "bytes":
            if set(value) != {"__bcpafl_type__", "base64"}:
                raise WireCodecError("malformed byte-string tag")
            try:
                return base64.b64decode(value["base64"], validate=True)
            except (binascii.Error, TypeError, ValueError) as exc:
                raise WireCodecError("invalid base64 byte string") from exc
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise WireCodecError("non-finite numbers are not allowed")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise WireCodecError("decoded unsupported JSON value")


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode a message mapping as compact UTF-8 JSON with tagged bytes."""
    if not isinstance(message, Mapping):
        raise WireCodecError("message must be a mapping")
    try:
        return json.dumps(
            _to_wire(message), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, WireCodecError):
            raise
        raise WireCodecError("message cannot be encoded") from exc


def decode_message(data: bytes) -> dict[str, Any]:
    """Decode a safe wire message, rejecting malformed or non-object JSON."""
    try:
        decoded = _from_wire(json.loads(data.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireCodecError("malformed JSON message") from exc
    if not isinstance(decoded, dict):
        raise WireCodecError("top-level message must be an object")
    return decoded

