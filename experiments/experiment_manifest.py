from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def write_experiment_manifest(
    manifest_path: Path,
    *,
    experiment_name: str,
    args: argparse.Namespace,
    outputs: Sequence[Path],
    archives: Sequence[Path] = (),
    root: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    root_path = root or Path(__file__).resolve().parents[1]
    payload = {
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(item) for item in sys.argv],
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "working_directory": str(Path.cwd()),
        "repository": repository_metadata(root_path),
        "arguments": to_jsonable(vars(args)),
        "inputs": {
            "archives": [path_metadata(path) for path in archives],
        },
        "outputs": [path_metadata(path) for path in outputs],
        "extra": to_jsonable(dict(extra or {})),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def repository_metadata(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "commit": git_value(root, "rev-parse", "HEAD"),
        "branch": git_value(root, "branch", "--show-current"),
        "remote_origin": git_value(root, "remote", "get-url", "origin"),
        "dirty": bool(git_value(root, "status", "--porcelain")),
    }


def path_metadata(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    exists = expanded.exists()
    return {
        "path": str(path),
        "absolute_path": str(expanded.resolve(strict=False)),
        "exists": exists,
        "size_bytes": expanded.stat().st_size if exists and expanded.is_file() else None,
    }


def git_value(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
