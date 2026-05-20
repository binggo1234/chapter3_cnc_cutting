from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from experiment_manifest import (  # noqa: E402
    default_manifest_path,
    to_jsonable,
    write_experiment_manifest,
)


def test_default_manifest_path_uses_output_stem() -> None:
    assert default_manifest_path(Path("results/routes.csv")) == Path(
        "results/routes_manifest.json"
    )


def test_to_jsonable_converts_paths_and_tuples() -> None:
    payload = to_jsonable(
        {
            "path": Path("results/out.csv"),
            "items": (Path("a"), 3),
        }
    )

    assert payload == {"path": "results/out.csv", "items": ["a", 3]}


def test_write_experiment_manifest_records_inputs_outputs(tmp_path: Path) -> None:
    archive = tmp_path / "input.zip"
    output = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.json"
    archive.write_text("input", encoding="utf-8")
    output.write_text("result", encoding="utf-8")

    write_experiment_manifest(
        manifest,
        experiment_name="unit_test",
        args=argparse.Namespace(output=output, methods=("greedy", "beam")),
        archives=(archive,),
        outputs=(output,),
        root=tmp_path,
        extra={"row_count": 2},
    )

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["experiment_name"] == "unit_test"
    assert data["arguments"]["methods"] == ["greedy", "beam"]
    assert data["inputs"]["archives"][0]["exists"] is True
    assert data["inputs"]["archives"][0]["absolute_path"] == str(archive)
    assert data["outputs"][0]["size_bytes"] == len("result")
    assert data["extra"]["row_count"] == 2
