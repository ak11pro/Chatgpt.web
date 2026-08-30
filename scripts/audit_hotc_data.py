#!/usr/bin/env python3
"""Create a reproducible availability/schema report before tracker inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hotc_tracker.data import discover_catalog  # noqa: E402

REQUIRED_COLUMNS = {"ID", "x", "y", "width", "height"}


def inspect_csv(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        rows = sum(1 for _ in reader)
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    return {"path": path, "rows": rows, "columns": columns, "missing_required_columns": missing, "schema_valid": not missing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="input root; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = discover_catalog(args.root or None)
    report = catalog.as_dict() | {"training": inspect_csv(catalog.training_csv), "sample_submission": inspect_csv(catalog.sample_submission_csv)}
    report["tracker_ready"] = bool(catalog.training_csv and catalog.frames_available)
    report["next_action"] = ("Implement and validate a FrameSource for the discovered raw-frame dataset." if report["tracker_ready"] else "Mount the organizer-approved raw HSI frame dataset; annotations alone cannot run a visual tracker.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
