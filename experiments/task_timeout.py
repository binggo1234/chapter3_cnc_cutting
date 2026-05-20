from __future__ import annotations

import signal
from contextlib import contextmanager
from types import FrameType
from typing import Iterator


class TaskTimeoutError(TimeoutError):
    pass


def timeout_supported() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "getitimer")
        and hasattr(signal, "setitimer")
    )


@contextmanager
def task_timeout(seconds: float | None) -> Iterator[None]:
    if seconds is None or seconds <= 0 or not timeout_supported():
        yield
        return

    def raise_timeout(signum: int, frame: FrameType | None) -> None:
        raise TaskTimeoutError(f"task exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
