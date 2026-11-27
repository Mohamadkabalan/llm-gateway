"""Ratio budgets over a sliding window: the retry budget and the hedge cap.

A budget allows ``max(minimum, ratio * requests)`` spends per window, counting the
requests noted in the same window. The window is a ring of one-second slots, so noting
a request or spending is O(1) and old traffic ages out smoothly. Budgets are per
process, which is what bounds the load one replica can add to a struggling provider.
"""

import math
import time
from collections.abc import Callable

_SLOTS = 10
_SLOT_SECONDS = 1.0


class RatioBudget:
    def __init__(
        self,
        ratio: float,
        minimum: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ratio = ratio
        self.minimum = minimum
        self._clock = clock
        self._requests = [0] * _SLOTS
        self._spent = [0] * _SLOTS
        self._epoch = [-1] * _SLOTS  # which second each slot currently counts

    def configure(self, ratio: float, minimum: int) -> None:
        """Pick up changed policy after a config reload, keeping the counts."""
        self.ratio, self.minimum = ratio, minimum

    def note_request(self) -> None:
        self._requests[self._slot()] += 1

    def try_spend(self) -> bool:
        slot = self._slot()
        requests = spent = 0
        now = self._second()
        for i in range(_SLOTS):
            if now - self._epoch[i] < _SLOTS:
                requests += self._requests[i]
                spent += self._spent[i]
        if spent >= max(self.minimum, math.floor(self.ratio * requests)):
            return False
        self._spent[slot] += 1
        return True

    def _second(self) -> int:
        return int(self._clock() // _SLOT_SECONDS)

    def _slot(self) -> int:
        second = self._second()
        slot = second % _SLOTS
        if self._epoch[slot] != second:
            self._epoch[slot] = second
            self._requests[slot] = 0
            self._spent[slot] = 0
        return slot
