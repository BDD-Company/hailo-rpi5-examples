#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from threading import Condition
import queue
import time
from typing import Generic, TypeVar

T = TypeVar("T")

class OverwriteQueue(Generic[T]):
    """Bounded queue with overwrite-oldest semantics (deque(maxsize))."""

    __slots__ = ("_d", "_cv")

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._d: deque[T] = deque(maxlen=maxsize)
        self._cv = Condition()

    def put(self, item: T) -> None:
        with self._cv:
            self._d.append(item)   # overwrites oldest when full
            self._cv.notify()      # 1:1 wakeup; use notify_all for many consumers

    def get(self, timeout: float | None = None) -> T:
        with self._cv:
            if self._d:
                return self._d.popleft()

            if timeout is None:
                while not self._d:
                    self._cv.wait()
                return self._d.popleft()

            deadline = time.monotonic() + timeout
            while not self._d:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cv.wait(remaining):
                    raise queue.Empty

            return self._d.popleft()

    def get_nowait(self) -> T:
        with self._cv:
            if not self._d:
                raise queue.Empty
            return self._d.popleft()

    def __len__(self) -> int:
        with self._cv:
            return len(self._d)

    def clear(self) -> None:
        with self._cv:
            self._d.clear()
