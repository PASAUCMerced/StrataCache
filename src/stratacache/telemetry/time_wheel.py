from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TimeWheelEvent:
    event: Any = None

class TimeWheel:
    """Non-blocking time wheel implementation
    Make sure only one thread is using the time wheel, otherwise you need to add locks to protect the internal state
    """
    def __init__(self, tick: int, slot_num: int, callback: callable, context: Any):
        self._tick = tick
        self._current_tick = 0
        self._slot_num = slot_num
        self._slots: dict[int, list[TimeWheelEvent]] = {i: [] for i in range(slot_num)}
        self._callback = callback
        self._context = context

    def add_event(self, event: Any, delay: float = 0.0) -> None:
        index = (self._current_tick + delay // self._tick) % self._slot_num
        self._slots[index].append(
            TimeWheelEvent(
                event=event
            )
        )

    def tick(self) -> None:
        self._run_slot()
        self._current_tick = (self._current_tick + 1) % self._slot_num

    def _run_slot(self) -> None:
        events = self._slots[self._current_tick]
        self._slots[self._current_tick] = []
        events = [e.event for e in events]
        if events:
            self._callback(events, self._context)