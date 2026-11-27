"""Circuit-breaker scenarios shared by the in-process and Redis store tests.

Each case is a list of steps at times relative to ``BASE`` (aligned to the 3 s buckets
of ``POLICY``'s 30 s window):

* ``("admit", t, admission, state)``: ``admit`` returns this admission and state;
* ``("record", t, ok, probe, state, transitioned)``: ``record`` returns this state.

Running one table against both stores is what keeps the Lua scripts and the Python
state machine equivalent.
"""

from typing import Literal

from switchboard.reliability.breaker import Admission, BreakerState
from switchboard.routing.config import BreakerPolicy

BASE = 1_764_000_000.0
POLICY = BreakerPolicy(
    window=30, min_requests=4, failure_rate=0.5, cooldown=10, max_cooldown=40, half_open_probes=2
)

A, P, D = Admission.ALLOW, Admission.PROBE, Admission.DENY
CLOSED, HALF, OPEN = BreakerState.CLOSED, BreakerState.HALF_OPEN, BreakerState.OPEN

type Admit = tuple[Literal["admit"], float, Admission, BreakerState]
type Record = tuple[Literal["record"], float, bool, bool, BreakerState, bool]
type Step = Admit | Record


def fail(t: float, state: BreakerState = CLOSED, transitioned: bool = False) -> Record:
    return ("record", t, False, False, state, transitioned)


def ok(t: float, state: BreakerState = CLOSED, transitioned: bool = False) -> Record:
    return ("record", t, True, False, state, transitioned)


def probe(t: float, success: bool, state: BreakerState, transitioned: bool) -> Record:
    return ("record", t, success, True, state, transitioned)


def admit(t: float, admission: Admission, state: BreakerState) -> Admit:
    return ("admit", t, admission, state)


# Four failures in a row open the breaker at t=3 (until t=13).
OPEN_AT_3: list[Step] = [fail(0), fail(1), fail(2), fail(3, OPEN, True)]

CASES: dict[str, list[Step]] = {
    "stays closed below min_requests": [
        fail(0),
        fail(1),
        fail(2),
        admit(3, A, CLOSED),
    ],
    "opens at the failure-rate threshold": [
        ok(0),
        ok(1),
        fail(2),
        fail(3, OPEN, True),  # 2 of 4 = 50%
        admit(4, D, OPEN),
    ],
    "stays closed under the threshold": [ok(0), ok(1), ok(2), fail(3), admit(4, A, CLOSED)],
    "outcomes older than the window age out": [
        fail(0),
        fail(1),
        fail(2),
        ok(45),  # the 0-3 s bucket has left the 30 s window
        fail(46),
        fail(47),
        admit(48, A, CLOSED),
        fail(48, OPEN, True),  # 4 outcomes in the window, 3 of them failures
    ],
    "half-open admits exactly the configured probes": [
        *OPEN_AT_3,
        admit(12.9, D, OPEN),
        admit(13, P, HALF),
        admit(13.1, P, HALF),
        admit(13.2, D, HALF),
        admit(13.3, D, HALF),
    ],
    "successful probes close it": [
        *OPEN_AT_3,
        admit(13, P, HALF),
        admit(13, P, HALF),
        probe(14, True, HALF, False),
        probe(15, True, CLOSED, True),
        admit(16, A, CLOSED),
        fail(16),  # a fresh window: one failure does not reopen it
        admit(17, A, CLOSED),
    ],
    "a failed probe reopens with a doubled cooldown": [
        *OPEN_AT_3,
        admit(13, P, HALF),
        probe(14, False, OPEN, True),  # open until 14 + 20
        admit(33.9, D, OPEN),
        admit(34, P, HALF),
        probe(35, False, OPEN, True),  # open until 35 + 40 (capped at max_cooldown)
        admit(74.9, D, OPEN),
        admit(75, P, HALF),
        probe(76, False, OPEN, True),  # still 40: the cap holds
        admit(115.9, D, OPEN),
        admit(116, P, HALF),
    ],
    "outcomes while open are ignored": [
        *OPEN_AT_3,
        ok(4, OPEN),
        fail(5, OPEN),
        admit(6, D, OPEN),
    ],
    "non-probe outcomes do not decide a half-open breaker": [
        *OPEN_AT_3,
        admit(13, P, HALF),
        ok(14, HALF),  # admitted before the breaker opened
        fail(14, HALF),
        admit(14, P, HALF),
        admit(14, D, HALF),
    ],
    "a probe round that never reports is restarted after a cooldown": [
        *OPEN_AT_3,
        admit(13, P, HALF),
        admit(13, P, HALF),
        admit(22.9, D, HALF),
        admit(23, P, HALF),  # 13 + cooldown: a new round
        admit(23, P, HALF),
        admit(23, D, HALF),
    ],
    # With SHORT (a 10 s window in 1 s buckets): failures spread across bucket
    # boundaries inside one window all count. (A ring of window-sized buckets once made
    # a 10 s window a tumbling one, so these never opened it.)
    "a short window slides across bucket boundaries": [
        fail(8.5),
        fail(9.5),
        fail(10.5),
        fail(11.5, OPEN, True),
    ],
    "a short window forgets what is older than it": [
        fail(0),
        fail(1),
        fail(2),
        fail(10.5),  # t=0 has left the window: 3 failures in it, under min_requests
        admit(10.6, A, CLOSED),
    ],
}

SHORT = BreakerPolicy(window=10, min_requests=4, failure_rate=0.5, cooldown=5)
CASE_POLICY = {name: SHORT for name in CASES if name.startswith("a short window")}


def policy_for(case: str) -> BreakerPolicy:
    return CASE_POLICY.get(case, POLICY)
