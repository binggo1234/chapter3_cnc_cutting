import json
import zipfile

from cnc_cutting.io import (
    find_related_config_member,
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_csv,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.models import Point


PLACEMENTS_CSV = """board,uid,pid_raw,x,y,w,h,w0,h0,rotated
1,101,101,5,5,100,50,100,50,0
1,102,102,105,5,80,50,80,50,0
2,201,201,5,5,60,40,60,40,0
"""


def test_load_chapter2_layouts_from_csv_groups_rows_by_board(tmp_path) -> None:
    path = tmp_path / "placements.csv"
    path.write_text(PLACEMENTS_CSV, encoding="utf-8")

    layouts = load_chapter2_layouts_from_csv(path, panel_width=2440, panel_height=1220)

    assert len(layouts) == 2
    assert layouts[0].panel_id == "placements_board_1"
    assert len(layouts[0].rectangles) == 2
    assert layouts[0].rectangles[0].part_id == "101"
    assert layouts[1].panel_id == "placements_board_2"
    assert len(layouts[1].rectangles) == 1


def test_find_related_config_member_handles_windows_style_members() -> None:
    placements = (
        r"root\case\method\seed_1000\placements.csv"
    )
    members = (
        r"root\case\config_dump.json",
        r"root\case\method\config_dump.json",
        placements,
    )

    assert find_related_config_member(members, placements) == r"root\case\method\config_dump.json"


def test_load_chapter2_layouts_from_zip_uses_related_config_and_board_filter(tmp_path) -> None:
    zip_path = tmp_path / "chapter2.zip"
    config = {
        "cfg": {
            "BOARD_W": 2440.0,
            "BOARD_H": 1220.0,
            "TRIM": 5.0,
            "TOOL_D": 6.0,
        }
    }
    placement_member = "root/case/method/seed_1000/placements.csv"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("root/case/method/config_dump.json", json.dumps(config))
        archive.writestr(placement_member, PLACEMENTS_CSV)

    layouts = load_chapter2_layouts_from_zip(
        zip_path,
        placements_member=placement_member,
        board_ids=(2,),
    )
    cfg = load_chapter2_config_from_zip(zip_path, placements_member=placement_member)
    tool = tool_config_from_chapter2_config(cfg)

    assert len(layouts) == 1
    assert layouts[0].panel_width == 2440
    assert layouts[0].panel_height == 1220
    assert layouts[0].panel_id.endswith("_board_2")
    assert len(layouts[0].rectangles) == 1
    assert tool.trim_margin == 5
    assert tool.tool_diameter == 6
    assert tool.centerline_boundary_margin == 3
    assert tool.allow_safe_lift_over_released_parts
    assert tool.allow_low_clearance_detour
    assert tool.start_point == Point(5, 5)
