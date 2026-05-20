from cnc_cutting.geometry import (
    segment_intersects_axis_aligned_rectangle,
    segment_intersects_polygon,
)
from cnc_cutting.models import Point


def test_segment_intersects_axis_aligned_rectangle_when_crossing() -> None:
    rectangle = (5.0, 5.0, 15.0, 15.0)

    assert segment_intersects_axis_aligned_rectangle(Point(0, 10), Point(20, 10), rectangle)


def test_segment_intersects_axis_aligned_rectangle_rejects_disjoint_bbox() -> None:
    rectangle = (5.0, 5.0, 15.0, 15.0)

    assert not segment_intersects_axis_aligned_rectangle(Point(0, 0), Point(4, 4), rectangle)


def test_segment_intersects_polygon_uses_rectangle_fast_path() -> None:
    polygon = (
        Point(5, 5),
        Point(15, 5),
        Point(15, 15),
        Point(5, 15),
    )

    assert segment_intersects_polygon(Point(0, 10), Point(20, 10), polygon)
    assert not segment_intersects_polygon(Point(0, 0), Point(4, 4), polygon)
