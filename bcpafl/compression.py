"""Model-update compression Omega_i in {0, Q8, Q4, Top-k} (Eq. 9).

"0" is interpreted as *no compression* (float32).  Every mode has a compact,
self-describing binary format, so payload sizes on the wire are the real
compressed sizes.  Lossy modes are paired with vehicle-side error feedback
(:class:`ErrorFeedback`) so the compression error of one round is re-added to
the next round's update instead of being lost.

Format::

    magic "BCPZ" | version u8 | mode u8 | n u32 | param u32 | body
    none : n float32
    q8/q4: per block of ``param`` values: scale float32, then int8 (q8) or
           packed signed nibbles in [-7, 7] (q4)
    topk : k = param; k uint32 indices then k float32 values
"""

from __future__ import annotations

import math
import struct

import numpy as np

MAGIC = b"BCPZ"
VERSION = 1
_MODES = {"none": 0, "q8": 1, "q4": 2, "topk": 3}
_MODE_NAMES = {v: k for k, v in _MODES.items()}
_HEADER = struct.Struct(">4sBBII")
BLOCK = 256


class CompressionError(ValueError):
    pass


def _stochastic_round(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    floor = np.floor(values)
    return floor + (rng.random(values.shape) < (values - floor))


def _quantize(vector: np.ndarray, levels: int, rng: np.random.Generator):
    n = vector.size
    n_blocks = math.ceil(n / BLOCK)
    padded = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    padded[:n] = vector
    blocks = padded.reshape(n_blocks, BLOCK)
    scales = np.abs(blocks).max(axis=1).astype(np.float32)
    safe = np.where(scales > 0, scales, 1.0)[:, None]
    q = _stochastic_round(blocks / safe * levels, rng)
    q = np.clip(q, -levels, levels).astype(np.int8)
    return scales, q


def compress(vector: np.ndarray, mode: str, *, topk_fraction: float = 0.1,
             rng: np.random.Generator | None = None) -> bytes:
    if mode not in _MODES:
        raise CompressionError(f"unknown compression mode {mode!r}")
    rng = rng or np.random.default_rng()
    vector = np.asarray(vector, dtype=np.float32).ravel()
    if not np.all(np.isfinite(vector)):
        raise CompressionError("cannot compress non-finite values")
    n = vector.size
    if mode == "none":
        return _HEADER.pack(MAGIC, VERSION, 0, n, 0) + vector.astype(">f4").tobytes()
    if mode in ("q8", "q4"):
        levels = 127 if mode == "q8" else 7
        scales, q = _quantize(vector, levels, rng)
        body = bytearray()
        for scale, row in zip(scales, q):
            body += struct.pack(">f", float(scale))
            if mode == "q8":
                body += row.astype(np.int8).tobytes()
            else:
                nib = (row.astype(np.int16) & 0x0F).astype(np.uint8)
                body += (nib[0::2] << 4 | nib[1::2]).astype(np.uint8).tobytes()
        return _HEADER.pack(MAGIC, VERSION, _MODES[mode], n, BLOCK) + bytes(body)
    k = max(1, int(math.ceil(topk_fraction * n)))
    idx = np.argpartition(np.abs(vector), n - k)[n - k:]
    idx.sort()
    return (_HEADER.pack(MAGIC, VERSION, 3, n, k) + idx.astype(">u4").tobytes()
            + vector[idx].astype(">f4").tobytes())


def decompress(payload: bytes, expected_size: int | None = None) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < _HEADER.size:
        raise CompressionError("payload too short")
    magic, version, mode, n, param = _HEADER.unpack_from(payload, 0)
    if magic != MAGIC or version != VERSION or mode not in _MODE_NAMES:
        raise CompressionError("bad compression header")
    if expected_size is not None and n != expected_size:
        raise CompressionError(f"payload has {n} values, expected {expected_size}")
    body = memoryview(payload)[_HEADER.size:]
    name = _MODE_NAMES[mode]
    if name == "none":
        if len(body) != 4 * n:
            raise CompressionError("truncated float32 payload")
        return np.frombuffer(body, dtype=">f4").astype(np.float32)
    if name in ("q8", "q4"):
        if param != BLOCK:
            raise CompressionError("unsupported block size")
        n_blocks = math.ceil(n / BLOCK)
        per_block = 4 + (BLOCK if name == "q8" else BLOCK // 2)
        if len(body) != n_blocks * per_block:
            raise CompressionError("truncated quantised payload")
        out = np.empty(n_blocks * BLOCK, dtype=np.float32)
        levels = 127 if name == "q8" else 7
        for b in range(n_blocks):
            chunk = body[b * per_block:(b + 1) * per_block]
            scale = struct.unpack(">f", chunk[:4])[0]
            if name == "q8":
                q = np.frombuffer(chunk[4:], dtype=np.int8).astype(np.float32)
            else:
                packed = np.frombuffer(chunk[4:], dtype=np.uint8)
                nib = np.empty(BLOCK, dtype=np.int16)
                nib[0::2] = packed >> 4
                nib[1::2] = packed & 0x0F
                nib = np.where(nib > 7, nib - 16, nib)
                q = nib.astype(np.float32)
            out[b * BLOCK:(b + 1) * BLOCK] = q / levels * scale
        result = out[:n]
        if not np.all(np.isfinite(result)):
            raise CompressionError("non-finite values in payload")
        return result
    k = param
    if not 1 <= k <= n or len(body) != 8 * k:
        raise CompressionError("truncated top-k payload")
    idx = np.frombuffer(body[:4 * k], dtype=">u4").astype(np.int64)
    vals = np.frombuffer(body[4 * k:], dtype=">f4").astype(np.float32)
    if idx.size and (idx.max() >= n or np.any(np.diff(idx) <= 0)):
        raise CompressionError("invalid top-k indices")
    out = np.zeros(n, dtype=np.float32)
    out[idx] = vals
    return out


def compressed_size(n: int, mode: str, topk_fraction: float = 0.1) -> int:
    """Exact payload size in bytes, used for the communication cost (Eq. 17)."""
    if mode == "none":
        return _HEADER.size + 4 * n
    if mode == "q8":
        return _HEADER.size + math.ceil(n / BLOCK) * (4 + BLOCK)
    if mode == "q4":
        return _HEADER.size + math.ceil(n / BLOCK) * (4 + BLOCK // 2)
    if mode == "topk":
        return _HEADER.size + 8 * max(1, int(math.ceil(topk_fraction * n)))
    raise CompressionError(f"unknown compression mode {mode!r}")


class ErrorFeedback:
    """Vehicle-side residual accumulator for lossy compression (EF-SGD)."""

    def __init__(self, size: int) -> None:
        self.residual = np.zeros(size, dtype=np.float32)

    def compress(self, delta: np.ndarray, mode: str, *, topk_fraction: float,
                 rng: np.random.Generator) -> bytes:
        corrected = delta.astype(np.float32) + self.residual
        payload = compress(corrected, mode, topk_fraction=topk_fraction, rng=rng)
        self.residual = corrected - decompress(payload, corrected.size)
        return payload

    def reset(self) -> None:
        self.residual[:] = 0.0
