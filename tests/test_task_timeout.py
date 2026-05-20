from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from task_timeout import TaskTimeoutError, task_timeout, timeout_supported  # noqa: E402


def test_task_timeout_disabled_allows_slow_block() -> None:
    with task_timeout(0):
        time.sleep(0.01)


def test_task_timeout_allows_fast_block() -> None:
    if not timeout_supported():
        pytest.skip("signal-based timeout is not supported on this platform")

    with task_timeout(0.1):
        time.sleep(0.01)


def test_task_timeout_raises_for_slow_block() -> None:
    if not timeout_supported():
        pytest.skip("signal-based timeout is not supported on this platform")

    with pytest.raises(TaskTimeoutError):
        with task_timeout(0.01):
            time.sleep(0.1)
