from cnc_cutting.models import PlacedRectangle
from cnc_cutting.path_cost import nearest_neighbor_order, total_air_move_distance


def test_nearest_neighbor_order_visits_each_rectangle_once() -> None:
    rectangles = (
        PlacedRectangle("A", 0, 0, 10, 10),
        PlacedRectangle("B", 100, 0, 10, 10),
        PlacedRectangle("C", 50, 0, 10, 10),
    )

    order = nearest_neighbor_order(rectangles)

    assert sorted(order) == [0, 1, 2]


def test_total_air_move_distance_for_known_order() -> None:
    rectangles = (
        PlacedRectangle("A", 3, 4, 10, 10),
        PlacedRectangle("B", 6, 8, 10, 10),
    )

    distance = total_air_move_distance(rectangles, (0, 1))

    assert distance == 10.0

