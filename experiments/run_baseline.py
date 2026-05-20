from __future__ import annotations

from pathlib import Path

from cnc_cutting.io import load_layout
from cnc_cutting.models import CuttingConfig
from cnc_cutting.path_cost import nearest_neighbor_order, total_air_move_distance


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    layout = load_layout(ROOT / "data" / "sample_layouts" / "demo_layout.json")
    order = nearest_neighbor_order(layout.rectangles)
    distance = total_air_move_distance(layout.rectangles, order, CuttingConfig())

    print(f"panel_id: {layout.panel_id}")
    print(f"rectangle_count: {len(layout.rectangles)}")
    print(f"nearest_neighbor_order: {order}")
    print(f"air_move_distance: {distance:.3f}")


if __name__ == "__main__":
    main()

