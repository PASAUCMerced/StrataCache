"""
MemoryObj: type-agnostic in-memory artifact handle.

Replaces the previous `bytes` payload type that flowed through the backend /
tiering / engine layers. A MemoryObj wraps the underlying buffer (Python
bytes today, pinned host tensor / device tensor in later phases) and carries
the metadata needed to reconstruct it.

Design intent (this phase):
- Provide a uniform protocol so backend.put/get exchange MemoryObjs instead
  of raw bytes.
- Keep two concrete implementations:
  * BytesMemoryObj: thin wrapper around Python bytes. Used to keep the CPU
    OrderedDict backend working with no copy-format change.
  * TensorMemoryObj: wraps a torch tensor (pinned or device). Backs the
    zero-copy slab path.
- Reserve, but do NOT yet implement, ref-count / pin / auto-free hooks.
  Those land together with the slab allocator in a later phase
  (see tmp/lmcache_gap_analysis.md, B1-B3).

Acknowledgement: shape and field names follow LMCache's MemoryObj /
TensorMemoryObj for future compatibility (Apache-2.0).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from stratacache.core.artifact import ArtifactMeta


@dataclass(slots=True)
class MemoryObjMetadata:
    """
    Per-buffer metadata that travels with the MemoryObj at runtime.

    `artifact_meta` is the user-facing JSON-serializable metadata
    (ArtifactType / engine_hints / attrs). The remaining fields are
    runtime-only book-keeping the storage plane uses.
    """

    artifact_meta: ArtifactMeta
    # Logical payload size in bytes. Allocators may round up internally;
    # `size` always reports the user-visible payload length.
    size: int
    # Optional dtype/shape hints when the buffer is a typed tensor.
    # None when the buffer is opaque bytes (e.g. an opaque CXL frame).
    dtype: Optional[str] = None
    shape: Optional[tuple[int, ...]] = None
    # B3 ref-count / pin protocol.
    # ref_count = 1 at construction (the storage layer holds one ref).
    # Increments when consumers borrow the object via get_blocking-style
    # paths; reaches 0 only after the layer has dropped its ref AND every
    # borrower has released theirs. pin_count tracks lookup-pins that
    # forbid eviction even when ref_count > 1.
    ref_count: int = 1
    pin_count: int = 0


class MemoryObj(ABC):
    """
    Abstract handle for an artifact's in-memory buffer.

    Backends accept MemoryObj on put and return MemoryObj on get. They MUST
    NOT assume any concrete subclass: a CPU backend may store a
    TensorMemoryObj, while a CXL backend serializes through the byte view.

    Subclasses provide either `tensor` (preferred for zero-copy paths) or
    `byte_array` (always available, may copy). Callers should consult
    `metadata.dtype` / `metadata.shape` to know whether `tensor` is usable.
    """

    @property
    @abstractmethod
    def metadata(self) -> MemoryObjMetadata: ...

    @property
    @abstractmethod
    def byte_array(self) -> bytes:
        """
        Return the buffer as a contiguous immutable bytes view.

        For BytesMemoryObj this is the underlying bytes (no copy). For
        TensorMemoryObj this materializes the tensor as raw uint8 bytes.
        """
        ...

    @property
    def tensor(self) -> Any:
        """
        Return the underlying tensor (torch.Tensor) if available, else None.

        Default: None. Subclasses backed by tensors override this.
        """
        return None

    def get_size(self) -> int:
        """Logical payload size in bytes (replaces `len(payload)`)."""
        return self.metadata.size

    # -- B3 ref-count / pin protocol ----------------------------------------

    def ref_count_up(self) -> None:
        self.metadata.ref_count += 1

    def ref_count_down(self) -> None:
        if self.metadata.ref_count > 0:
            self.metadata.ref_count -= 1
        if self.metadata.ref_count == 0 and self.metadata.pin_count == 0:
            self._on_release()

    def pin(self) -> None:
        self.metadata.pin_count += 1

    def unpin(self) -> None:
        if self.metadata.pin_count > 0:
            self.metadata.pin_count -= 1
        if self.metadata.ref_count == 0 and self.metadata.pin_count == 0:
            self._on_release()

    def can_evict(self) -> bool:
        """
        True iff the object is not pinned AND only the storage layer holds
        a reference (ref_count <= 1).
        """
        return self.metadata.pin_count == 0 and self.metadata.ref_count <= 1

    def _on_release(self) -> None:
        """
        Hook fired when ref_count and pin_count both reach 0.

        Default: no-op (BytesMemoryObj relies on Python GC). TensorMemoryObj
        with an allocator-backed slot overrides this to free the slot.
        """
        return None


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class BytesMemoryObj(MemoryObj):
    """
    Trivial MemoryObj backed by a Python bytes object.

    Used by the CPU backend during the migration phase; also returned by
    the CXL backend's read path when the caller did not supply a
    dtype/shape hint. Carries no dtype/shape information by default.
    """

    __slots__ = ("_buf", "_meta")

    def __init__(self, buf: bytes, artifact_meta: ArtifactMeta) -> None:
        self._buf = buf
        self._meta = MemoryObjMetadata(
            artifact_meta=artifact_meta,
            size=len(buf),
        )

    @classmethod
    def from_bytes(cls, buf: bytes, artifact_meta: ArtifactMeta) -> "BytesMemoryObj":
        return cls(buf, artifact_meta)

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self._meta

    @property
    def byte_array(self) -> bytes:
        return self._buf


class TensorMemoryObj(MemoryObj):
    """
    MemoryObj backed by a torch.Tensor.

    The tensor's underlying storage may be:
    - a pageable CPU tensor (simple case)
    - a pinned host tensor (when produced by a slab allocator with
      pin_memory=True)
    - a device tensor (when used as an in-flight GPU buffer)

    `release_callback`, when provided, is invoked exactly once when the
    object's ref_count and pin_count both reach 0. Slab-backed factories
    use this to return the slot to the allocator without the core layer
    needing to know about allocator types.

    `byte_array` materializes by viewing as uint8 then `.tobytes()`. Avoid
    using it on the hot path; prefer `.tensor` for zero-copy consumers.
    """

    __slots__ = (
        "_t",
        "_meta",
        "_release_cb",
        "_released",
        "_pending_event",
    )

    def __init__(
        self,
        tensor: Any,  # torch.Tensor; kept Any for safe import
        artifact_meta: ArtifactMeta,
        *,
        size: Optional[int] = None,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
        release_callback: Optional[Any] = None,  # Callable[[], None]
    ) -> None:
        self._t = tensor
        # Optional CUDA event recorded by producers of pending D2H copies.
        # Consumers that need host-visible bytes (or want to chain an H2D
        # back to GPU) call `wait_pending()` first - cheap when the copy
        # has already drained, avoids the host-block in the producer path.
        self._pending_event: Any = None
        # Compute size lazily-but-eagerly so backends can account for it
        # without importing torch.
        if size is None:
            size = int(tensor.numel() * tensor.element_size())
        if dtype is None:
            dtype = str(tensor.dtype).replace("torch.", "")
        if shape is None:
            shape = tuple(int(x) for x in tensor.shape)
        self._meta = MemoryObjMetadata(
            artifact_meta=artifact_meta,
            size=int(size),
            dtype=dtype,
            shape=shape,
        )
        self._release_cb = release_callback
        self._released = False

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self._meta

    @property
    def tensor(self) -> Any:
        return self._t

    def wait_pending_on_stream(self, stream: Any = None) -> None:
        """
        GPU-side wait: makes `stream` (current stream by default) wait for
        the producer's pending copy event. NOT host-blocking. Safe to
        call repeatedly; doesn't clear the event so other consumers can
        also wait.

        Use this when the consumer immediately issues another GPU op that
        reads the buffer (e.g. an H2D copy back to GPU). Cheap when the
        producer ran on the same stream - in that case it's effectively
        a no-op.
        """
        ev = self._pending_event
        if ev is None:
            return
        try:
            ev.wait(stream)
        except TypeError:
            # Older torch: wait() takes no args; just no-op host wait.
            ev.wait()

    def wait_pending_on_host(self) -> None:
        """
        Host-blocking wait. Use only when the consumer is going to read
        the underlying memory from CPU (e.g. byte_array, CXL serialize).
        Clears the event after waiting.
        """
        ev = self._pending_event
        if ev is not None:
            ev.synchronize()
            self._pending_event = None

    def attach_pending_event(self, event: Any) -> None:
        """Producer-side hook to register a CUDA event."""
        self._pending_event = event

    @property
    def byte_array(self) -> bytes:
        # Lazy import keeps core dependency-free when torch isn't installed.
        import torch  # type: ignore[import-not-found]

        # CPU-side read: must drain any pending D2H from the producer.
        self.wait_pending_on_host()
        t = self._t.detach().contiguous()
        if t.device.type != "cpu":
            t = t.cpu()
        return t.view(torch.uint8).numpy().tobytes(order="C")

    def _on_release(self) -> None:
        if self._released:
            return
        self._released = True
        cb = self._release_cb
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                # Releasing memory must never raise to callers.
                pass


__all__ = [
    "MemoryObj",
    "MemoryObjMetadata",
    "BytesMemoryObj",
    "TensorMemoryObj",
]
