from __future__ import annotations


def run() -> None:
    """
    A fake backend that records the dtype/shape hints it receives on get,
    proving the chain + engine threads them through.
    """
    from stratacache.backend.base import BackendStats, MemoryLayer
    from stratacache.core.artifact import ArtifactId, ArtifactMeta
    from stratacache.core.errors import ArtifactNotFound
    from stratacache.core.memory_obj import BytesMemoryObj, MemoryObj
    from stratacache.engine import StorageEngine

    class HintRecorder(MemoryLayer):
        def __init__(self, store_name="hint"):
            self._name = store_name
            self._d: dict[str, MemoryObj] = {}
            self.last_get_hint = None  # (dtype, shape)

        @property
        def name(self):
            return self._name

        def exists(self, aid):
            return str(aid) in self._d

        def get(self, aid, *, dtype=None, shape=None):
            self.last_get_hint = (dtype, shape)
            mo = self._d.get(str(aid))
            if mo is None:
                raise ArtifactNotFound(str(aid))
            return mo

        def put(self, aid, mo):
            old = self._d.pop(str(aid), None)
            self._d[str(aid)] = mo
            return old.get_size() if old is not None else 0

        def delete(self, aid):
            mo = self._d.pop(str(aid), None)
            return mo.get_size() if mo is not None else 0

        def stats(self):
            return BackendStats(items=len(self._d), bytes_used=0)

    layer = HintRecorder()
    eng = StorageEngine.from_tiers(tiers=[layer], links=[], enable_writeback_worker=False)
    try:
        aid = ArtifactId("x")
        eng.store(aid, BytesMemoryObj(b"abc", ArtifactMeta()))

        # No hint.
        eng.load(aid)
        assert layer.last_get_hint == (None, None)

        # With hint.
        eng.load(aid, dtype="bfloat16", shape=(2, 3, 4))
        assert layer.last_get_hint == ("bfloat16", (2, 3, 4))
    finally:
        eng.close()
