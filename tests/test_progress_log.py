from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from progress_log import (  # noqa: E402
    append_progress_event,
    default_progress_log_path,
    new_progress_event,
    prepare_progress_log,
)


def test_default_progress_log_path_uses_output_stem() -> None:
    assert default_progress_log_path(Path("results/routes.csv")) == Path(
        "results/routes_progress.csv"
    )


def test_append_progress_event_writes_header_and_rows(tmp_path: Path) -> None:
    output = tmp_path / "progress.csv"

    append_progress_event(
        output,
        new_progress_event(
            event="started",
            archive="input.zip",
            placements_member="case/placements.csv",
            board_id="3",
            method="process_aware_beam",
            rectangle_count=12,
            candidate_unit_count=65,
        ),
    )
    append_progress_event(
        output,
        new_progress_event(
            event="completed",
            archive="input.zip",
            placements_member="case/placements.csv",
            board_id="3",
            method="process_aware_beam",
            rectangle_count=12,
            candidate_unit_count=65,
            elapsed_ms=123.4,
        ),
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["event"] for row in rows] == ["started", "completed"]
    assert rows[0]["rectangle_count"] == "12"
    assert rows[1]["elapsed_ms"] == "123.4"


def test_prepare_progress_log_preserves_file_when_resuming(tmp_path: Path) -> None:
    output = tmp_path / "progress.csv"
    output.write_text("existing", encoding="utf-8")

    prepare_progress_log(output, resume=True)
    assert output.read_text(encoding="utf-8") == "existing"

    prepare_progress_log(output, resume=False)
    assert output.read_text(encoding="utf-8") == ""
