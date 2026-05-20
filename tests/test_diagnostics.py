from cnc_cutting.diagnostics import diagnose_actions
from cnc_cutting.models import CuttingAction, CuttingActionType, Panel, Point, ToolConfig


def test_diagnose_actions_reports_boundary_penalty() -> None:
    action = CuttingAction(CuttingActionType.CUT, Point(1, 10), Point(10, 10), "s1")

    diagnostics = diagnose_actions(
        (action,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=5, tool_diameter=6),
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].action_index == 0
    assert diagnostics[0].action_type == "cut"
    assert diagnostics[0].boundary_penalty == 1.0
    assert diagnostics[0].collision_penalty == 0.0
    assert diagnostics[0].stability_penalty_delta == 0.0
    assert diagnostics[0].unstable_parts_before == 0
    assert diagnostics[0].unstable_parts_after == 0
