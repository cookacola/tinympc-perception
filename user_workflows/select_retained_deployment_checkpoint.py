#!/usr/bin/env python3
"""Select a safety-first ESPNet teacher without consulting test metrics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--minimum-unsafe-recall", type=float, default=0.963)
    parser.add_argument("--minimum-collision-ap", type=float, default=0.90)
    parser.add_argument("--maximum-real-corner-error", type=float, default=20.0)
    parser.add_argument("--maximum-negative-mask-pixel-rate", type=float, default=0.005)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    feasible = []
    for record in summary["history"]:
        obstacle = record["obstacle_validation"]
        gate = record["gate_validation"]
        if (
            obstacle["unsafe_clearance_recall"] >= args.minimum_unsafe_recall
            and obstacle["collision_ap"] >= args.minimum_collision_ap
            and gate["real_corner_mean_px"] <= args.maximum_real_corner_error
            and gate["explicit_no_gate_mask_pixel_rate"]
            <= args.maximum_negative_mask_pixel_rate
        ):
            feasible.append(record)
    if not feasible:
        raise RuntimeError("no checkpoint satisfies deployment constraints")
    # Safety is lexicographically first. The test split remains untouched.
    selected = max(
        feasible,
        key=lambda record: (
            record["obstacle_validation"]["unsafe_clearance_recall"],
            record["obstacle_validation"]["collision_ap"],
            record["gate_validation"]["synthetic_gate_iou"],
        ),
    )
    source = args.run / ("epoch_%03d.pt" % selected["epoch"])
    destination = args.run / "selected_for_deployment.pt"
    shutil.copy2(source, destination)
    report = {
        "selected_epoch": selected["epoch"],
        "selection_split": "validation",
        "test_metrics_used_for_selection": False,
        "ranking": "unsafe recall, then collision AP, then synthetic gate IoU",
        "constraints": {
            "minimum_unsafe_recall": args.minimum_unsafe_recall,
            "minimum_collision_ap": args.minimum_collision_ap,
            "maximum_real_corner_error": args.maximum_real_corner_error,
            "maximum_negative_mask_pixel_rate": args.maximum_negative_mask_pixel_rate,
        },
        "selected_record": selected,
        "feasible_epochs": [record["epoch"] for record in feasible],
        "checkpoint": str(destination),
    }
    (args.run / "deployment_selection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
