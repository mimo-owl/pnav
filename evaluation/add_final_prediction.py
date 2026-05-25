"""
Re-parse raw_response from all_runs_long.csv using a "last x,y JSON" strategy
and add *_final columns alongside the existing (current) columns.

Current strategy  : last JSON that has ALL required keys (selected_image+x+y for A/B,
                    x+y for C) AND is in-range; falls back to last with all keys if
                    none are in-range.  For A/B self-corrections that drop selected_image
                    this effectively falls back to the first full JSON.

Final strategy    : last JSON that has x AND y (selected_image not required).
                    Coordinates are clamped to [0, W-1] x [0, H-1].

New columns added:
    predicted_x_final, predicted_y_final
    predicted_x_norm_final, predicted_y_norm_final
    error_l2_final
    reparse_note   ("ok", "no_xy_json", "no_raw_response")

Usage:
    python evaluation/scripts/add_final_prediction.py \
        --csv     analysis/all_runs_long.csv \
        --output  analysis/all_runs_long_with_final.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

SCREEN_WIDTH  = 512
SCREEN_HEIGHT = 512


def _extract_all_jsons(text: str) -> list[dict]:
    results = []
    for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            results.append(json.loads(m.group()))
        except (ValueError, TypeError):
            pass
    return results


def _parse_final(raw: str, w: int = SCREEN_WIDTH, h: int = SCREEN_HEIGHT) -> tuple[float, float, str] | None:
    """
    Pick the last JSON that contains both 'x' and 'y'.
    Coordinates are clamped to valid range.
    Returns (x_px, y_px, note) or None.
    """
    candidates = _extract_all_jsons(raw)
    for obj in reversed(candidates):
        if "x" not in obj or "y" not in obj:
            continue
        try:
            x = float(obj["x"])
            y = float(obj["y"])
        except (ValueError, TypeError):
            continue
        x = max(0.0, min(float(w - 1), x))
        y = max(0.0, min(float(h - 1), y))
        return x, y, "ok"
    return None


def process(csv_path: Path, output_path: Path) -> None:
    print(f"Reading {csv_path} ...")
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames or []
        rows = list(reader)

    print(f"  {len(rows)} rows, {len(orig_fields)} columns")

    new_fields = [
        "predicted_x_final", "predicted_y_final",
        "predicted_x_norm_final", "predicted_y_norm_final",
        "error_l2_final",
        "reparse_note",
    ]

    n_ok = n_no_raw = n_no_json = 0

    for r in rows:
        raw = r.get("raw_response", "").strip()

        if not raw:
            r["predicted_x_final"]      = ""
            r["predicted_y_final"]      = ""
            r["predicted_x_norm_final"] = ""
            r["predicted_y_norm_final"] = ""
            r["error_l2_final"]         = ""
            r["reparse_note"]           = "no_raw_response"
            n_no_raw += 1
            continue

        result = _parse_final(raw)

        if result is None:
            r["predicted_x_final"]      = ""
            r["predicted_y_final"]      = ""
            r["predicted_x_norm_final"] = ""
            r["predicted_y_norm_final"] = ""
            r["error_l2_final"]         = ""
            r["reparse_note"]           = "no_xy_json"
            n_no_json += 1
            continue

        x_px, y_px, _ = result
        x_norm = x_px / SCREEN_WIDTH
        y_norm = y_px / SCREEN_HEIGHT

        r["predicted_x_final"]      = f"{x_px:.4f}"
        r["predicted_y_final"]      = f"{y_px:.4f}"
        r["predicted_x_norm_final"] = f"{x_norm:.6f}"
        r["predicted_y_norm_final"] = f"{y_norm:.6f}"

        gt_x = r.get("gt_x", "")
        gt_y = r.get("gt_y", "")
        if gt_x != "" and gt_y != "":
            try:
                ex = x_norm - float(gt_x)
                ey = y_norm - float(gt_y)
                r["error_l2_final"] = f"{math.sqrt(ex**2 + ey**2):.6f}"
            except (ValueError, TypeError):
                r["error_l2_final"] = ""
        else:
            r["error_l2_final"] = ""

        r["reparse_note"] = "ok"
        n_ok += 1

    print(f"  reparse ok={n_ok}  no_raw={n_no_raw}  no_json={n_no_json}")

    all_fields = orig_fields + new_fields
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    kb = output_path.stat().st_size // 1024
    print(f"Saved: {output_path}  ({kb} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True, help="Path to all_runs_long.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    process(Path(args.csv), Path(args.output))


if __name__ == "__main__":
    main()
