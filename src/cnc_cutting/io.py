from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile
from collections.abc import Iterable

from .models import Layout, PlacedRectangle, Point, ToolConfig


CHAPTER2_PLACEMENT_COLUMNS = frozenset({"board", "uid", "x", "y", "w", "h"})


def load_layout(path: str | Path) -> Layout:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rectangles = tuple(
        PlacedRectangle(
            part_id=str(item["part_id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            width=float(item["width"]),
            height=float(item["height"]),
        )
        for item in data["rectangles"]
    )
    return Layout(
        panel_id=str(data["panel_id"]),
        panel_width=float(data["panel_width"]),
        panel_height=float(data["panel_height"]),
        rectangles=rectangles,
    )


def discover_chapter2_placement_members(zip_path: str | Path) -> tuple[str, ...]:
    """Return placement CSV members from a chapter-2 experiment archive."""

    with zipfile.ZipFile(zip_path) as archive:
        return tuple(
            name
            for name in archive.namelist()
            if name.endswith("placements.csv") and not name.endswith("/")
        )


def find_related_config_member(
    members: Iterable[str],
    placements_member: str,
) -> str | None:
    """Find the nearest config_dump.json for a placement CSV inside a zip archive."""

    member_set = set(members)
    for separator in ("/", "\\"):
        if separator not in placements_member:
            continue
        parts = placements_member.split(separator)
        for index in range(len(parts) - 1, 0, -1):
            ancestor = separator.join(parts[:index])
            candidate = f"{ancestor}{separator}config_dump.json"
            if candidate in member_set:
                return candidate
    return None


def load_chapter2_config_from_zip(
    zip_path: str | Path,
    placements_member: str | None = None,
    config_member: str | None = None,
) -> dict:
    """Load the cfg payload associated with a chapter-2 placement result."""

    with zipfile.ZipFile(zip_path) as archive:
        if config_member is None:
            if placements_member is None:
                config_members = [
                    name for name in archive.namelist() if name.endswith("config_dump.json")
                ]
                if not config_members:
                    raise ValueError(f"no config_dump.json found in {zip_path}")
                config_member = config_members[0]
            else:
                config_member = find_related_config_member(
                    archive.namelist(),
                    placements_member,
                )
                if config_member is None:
                    raise ValueError(
                        f"no related config_dump.json found for {placements_member}"
                    )
        data = json.loads(archive.read(config_member).decode("utf-8-sig"))
    return data.get("cfg", data)


def tool_config_from_chapter2_config(
    config: dict,
    use_stock_boundary_margin: bool = True,
) -> ToolConfig:
    """Build a cutting ToolConfig from chapter-2 nesting parameters.

    Chapter-2 placements already include the trimming offset in x/y coordinates.
    For those imported layouts, the hard panel-boundary check should only keep
    the cutter center inside the stock panel instead of adding the trim margin
    a second time.
    """

    trim_margin = float(config.get("TRIM", 5.0))
    tool_diameter = float(config.get("TOOL_D", 6.0))
    tool_radius = tool_diameter / 2.0
    boundary_margin = tool_radius if use_stock_boundary_margin else None
    return ToolConfig(
        trim_margin=trim_margin,
        tool_diameter=tool_diameter,
        centerline_boundary_margin=boundary_margin,
        allow_safe_lift_over_released_parts=True,
        allow_low_clearance_detour=True,
        start_point=Point(trim_margin, trim_margin),
    )


def load_chapter2_layouts_from_csv(
    path: str | Path,
    panel_width: float,
    panel_height: float,
    board_ids: Iterable[int | str] | None = None,
    panel_id_prefix: str | None = None,
) -> tuple[Layout, ...]:
    """Load one or more single-board layouts from a chapter-2 placements.csv."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return _layouts_from_chapter2_rows(
            reader,
            panel_width=panel_width,
            panel_height=panel_height,
            board_ids=board_ids,
            panel_id_prefix=panel_id_prefix or Path(path).stem,
        )


def load_chapter2_layouts_from_zip(
    zip_path: str | Path,
    placements_member: str | None = None,
    config_member: str | None = None,
    board_ids: Iterable[int | str] | None = None,
    panel_width: float | None = None,
    panel_height: float | None = None,
    panel_id_prefix: str | None = None,
) -> tuple[Layout, ...]:
    """Load chapter-2 placement results from a zip archive as single-board layouts."""

    with zipfile.ZipFile(zip_path) as archive:
        if placements_member is None:
            placement_members = [
                name
                for name in archive.namelist()
                if name.endswith("placements.csv") and not name.endswith("/")
            ]
            if not placement_members:
                raise ValueError(f"no placements.csv found in {zip_path}")
            placements_member = placement_members[0]

        if panel_width is None or panel_height is None:
            if config_member is None:
                config_member = find_related_config_member(
                    archive.namelist(),
                    placements_member,
                )
            if config_member is None:
                raise ValueError(
                    "panel_width and panel_height are required when no related "
                    "config_dump.json is available"
                )
            config = json.loads(archive.read(config_member).decode("utf-8-sig"))
            cfg = config.get("cfg", config)
            panel_width = float(cfg["BOARD_W"])
            panel_height = float(cfg["BOARD_H"])

        with archive.open(placements_member) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_handle)
            return _layouts_from_chapter2_rows(
                reader,
                panel_width=panel_width,
                panel_height=panel_height,
                board_ids=board_ids,
                panel_id_prefix=panel_id_prefix or _safe_panel_id(placements_member),
            )


def _layouts_from_chapter2_rows(
    rows: Iterable[dict[str, str]],
    panel_width: float,
    panel_height: float,
    board_ids: Iterable[int | str] | None,
    panel_id_prefix: str,
) -> tuple[Layout, ...]:
    requested_board_ids = (
        {_canonical_board_id(board_id) for board_id in board_ids}
        if board_ids is not None
        else None
    )
    rectangles_by_board: dict[str, list[PlacedRectangle]] = {}
    checked_columns = False

    for row_index, row in enumerate(rows):
        if not checked_columns:
            missing_columns = CHAPTER2_PLACEMENT_COLUMNS - set(row)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"placements.csv missing required columns: {missing}")
            checked_columns = True

        board_id = _canonical_board_id(row["board"])
        if requested_board_ids is not None and board_id not in requested_board_ids:
            continue
        part_id = str(row.get("uid") or row.get("pid_raw") or row_index)
        rectangles_by_board.setdefault(board_id, []).append(
            PlacedRectangle(
                part_id=part_id,
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["w"]),
                height=float(row["h"]),
            )
        )

    if requested_board_ids is not None:
        missing_boards = requested_board_ids - set(rectangles_by_board)
        if missing_boards:
            missing = ", ".join(sorted(missing_boards, key=_board_sort_key))
            raise ValueError(f"requested board_id not found in placements.csv: {missing}")

    return tuple(
        Layout(
            panel_id=f"{panel_id_prefix}_board_{board_id}",
            panel_width=panel_width,
            panel_height=panel_height,
            rectangles=tuple(rectangles_by_board[board_id]),
        )
        for board_id in sorted(rectangles_by_board, key=_board_sort_key)
    )


def _canonical_board_id(value: int | str) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _board_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _safe_panel_id(member: str) -> str:
    safe_chars = [char if char.isalnum() else "_" for char in member]
    safe_name = "".join(safe_chars).strip("_")
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return safe_name
