from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ProgressEvent:
    event_time_utc: str
    event: str
    archive: str
    placements_member: str
    board_id: str
    method: str
    rectangle_count: int | None
    candidate_unit_count: int | None
    elapsed_ms: float | None
    message: str = ""


def default_progress_log_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_progress.csv")


def prepare_progress_log(output_path: Path, resume: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")


def new_progress_event(
    *,
    event: str,
    archive: str,
    placements_member: str,
    board_id: str,
    method: str,
    rectangle_count: int | None = None,
    candidate_unit_count: int | None = None,
    elapsed_ms: float | None = None,
    message: str = "",
) -> ProgressEvent:
    return ProgressEvent(
        event_time_utc=datetime.now(timezone.utc).isoformat(),
        event=event,
        archive=archive,
        placements_member=placements_member,
        board_id=board_id,
        method=method,
        rectangle_count=rectangle_count,
        candidate_unit_count=candidate_unit_count,
        elapsed_ms=elapsed_ms,
        message=message,
    )


def append_progress_event(output_path: Path, event: ProgressEvent) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field.name for field in fields(ProgressEvent)],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(event))
