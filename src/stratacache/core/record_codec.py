"""
Optional record framing utility (magic + meta-json + payload).

Currently unused by the storage plane: the CXL backend stores raw payload
only and reconstructs schema from caller-supplied dtype/shape hints (see
CxlMemoryLayer + the connector's KV template). Kept here as a small
self-contained codec in case a future backend (e.g. local disk) needs
self-describing on-disk records.
"""
from __future__ import annotations

import json
import struct
from typing import Tuple

from stratacache.core.artifact import ArtifactMeta
from stratacache.core.errors import CodecError


_MAGIC = b"SC01"
_HDR_STRUCT = struct.Struct("<4sI")  # magic, meta_len


def encode_record(payload: bytes, meta: ArtifactMeta) -> bytes:
    """
    Encode (meta, payload) into a single bytes record.

    Format:
      - 4 bytes magic: b"SC01"
      - 4 bytes little-endian uint32: meta_len
      - meta_len bytes: UTF-8 JSON of ArtifactMeta
      - remaining: payload bytes
    """
    meta_json = json.dumps(meta.to_json(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(meta_json) > 0xFFFFFFFF:
        raise CodecError("meta too large")
    return _HDR_STRUCT.pack(_MAGIC, len(meta_json)) + meta_json + payload


def decode_record(buf: bytes) -> Tuple[bytes, ArtifactMeta]:
    if len(buf) < _HDR_STRUCT.size:
        raise CodecError("record too short")
    magic, meta_len = _HDR_STRUCT.unpack_from(buf, 0)
    if magic != _MAGIC:
        raise CodecError(f"bad magic: {magic!r}")
    meta_start = _HDR_STRUCT.size
    meta_end = meta_start + int(meta_len)
    if meta_end > len(buf):
        raise CodecError("meta length out of bounds")
    try:
        meta_dict = json.loads(buf[meta_start:meta_end].decode("utf-8"))
    except Exception as e:  # noqa: BLE001 (v0.1: keep dependency-free)
        raise CodecError(f"failed to decode meta json: {e}") from e
    payload = buf[meta_end:]
    return payload, ArtifactMeta.from_json(meta_dict)

