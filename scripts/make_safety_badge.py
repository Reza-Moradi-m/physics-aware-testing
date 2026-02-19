#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GRADE_COLORS = {
    "A": "brightgreen",
    "B": "green",
    "C": "yellow",
    "D": "orange",
    "F": "red",
}


def parse_grade_from_report(report_text: str) -> str:
    """
    Expected line in report like:
      Safety Grade: **F** (worst effective accuracy=0.0000)
    """
    m = re.search(r"Safety Grade:\s*\*\*([A-F])\*\*", report_text)
    if not m:
        # fallback: try plain "Safety Grade: F"
        m = re.search(r"Safety Grade:\s*([A-F])", report_text)
    return m.group(1) if m else "?"


def build_badge_json(grade: str) -> dict:
    color = GRADE_COLORS.get(grade, "lightgrey")
    label = "Safety Grade"
    message = grade
    return {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to results/week06_report.md")
    ap.add_argument("--out", required=True, help="Output path for badge JSON")
    args = ap.parse_args()

    report_path = Path(args.report)
    out_path = Path(args.out)

    if not report_path.exists():
        # If report doesn't exist, produce an "unknown" badge instead of crashing CI
        badge = build_badge_json("?")
    else:
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        grade = parse_grade_from_report(text)
        badge = build_badge_json(grade)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(badge), encoding="utf-8")


if __name__ == "__main__":
    main()