from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from progress_bar import TerminalProgressBar, format_progress_line  # noqa: E402


def test_format_progress_line_includes_counts_and_percent() -> None:
    line = format_progress_line(3, 12, "completed", "board=2", width=12)

    assert "[###---------]" in line
    assert "3/12" in line
    assert "25.0%" in line
    assert "completed board=2" in line


def test_format_progress_line_clamps_current_to_total() -> None:
    line = format_progress_line(15, 10, "completed", width=10)

    assert "[##########]" in line
    assert "10/10" in line
    assert "100.0%" in line


def test_terminal_progress_bar_advances_even_when_disabled(capsys) -> None:
    bar = TerminalProgressBar(2, enabled=False)

    bar.start("tasks")
    bar.advance("completed", "first")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert bar.current == 1
