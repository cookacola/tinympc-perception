#!/usr/bin/env python3
"""Render held-out critical-TTC classifications from a DORY checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_dory_model import load_dory_checkpoint


def labeled(panel, text):
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(output, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def update(candidates, name, score, index, details, prefer_high=True):
    prior = candidates.get(name)
    if prior is None or (score > prior["score"] if prefer_high else score < prior["score"]):
        candidates[name] = {"score": float(score), "dataset_index": int(index), **details}


def scan(model, dataset, device, batch_size, workers, threshold):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    candidates, offset = {}, 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["images"].to(device, non_blocking=True),
                batch["onboard_state"].to(device, non_blocking=True),
            )
            probability = torch.softmax(output["risk_logits"], 1)[:, 2].cpu()
            valid = batch["ttc_valid"].bool().squeeze(1)
            truth = (
                batch["ttc_approaching"].bool().squeeze(1)
                & (batch["inverse_ttc"].squeeze(1) >= 2.0)
                & valid
            )
            prediction = (probability >= threshold) & valid
            for local in range(probability.shape[0]):
                valid_count = max(int(valid[local].sum()), 1)
                truth_count = int(truth[local].sum())
                predicted_count = int(prediction[local].sum())
                tp = int((truth[local] & prediction[local]).sum())
                fp = int((~truth[local] & prediction[local] & valid[local]).sum())
                fn = truth_count - tp
                recall = tp / max(truth_count, 1)
                precision = tp / max(tp + fp, 1)
                details = {
                    "truth_critical_pixels": truth_count,
                    "predicted_critical_pixels": predicted_count,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp, "fp": fp, "fn": fn,
                }
                index = offset + local
                if truth_count >= 20:
                    update(candidates, "true_positive", recall + precision, index, details)
                    update(candidates, "critical_miss", recall, index, details, False)
                    update(candidates, "dense_critical", truth_count / valid_count, index, details)
                else:
                    update(candidates, "safe_correct", predicted_count / valid_count,
                           index, details, False)
                    update(candidates, "false_alarm", fp / valid_count, index, details)
            offset += probability.shape[0]
    return candidates


def render(model, dataset, name, candidate, device, threshold, output_dir):
    sample = dataset[candidate["dataset_index"]]
    with torch.no_grad():
        output = model(
            sample["images"][None].to(device), sample["onboard_state"][None].to(device)
        )
    inv_prediction = output["inverse_ttc"][0, 0].cpu().numpy()
    critical_probability = torch.softmax(output["risk_logits"], 1)[0, 2].cpu().numpy()
    inv_truth = sample["inverse_ttc"][0].numpy()
    valid = sample["ttc_valid"][0].numpy().astype(bool)
    truth_critical = (
        sample["ttc_approaching"][0].numpy().astype(bool) & (inv_truth >= 2.0) & valid
    )
    predicted_critical = (critical_probability >= threshold) & valid
    image = np.rint(sample["images"][1].numpy() * 255).astype(np.uint8)
    base = cv2.cvtColor(cv2.resize(image, (320, 320), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)
    scale = max(float(np.percentile(inv_truth[valid], 99)) if valid.any() else 1.0, 2.0)
    truth_map = cv2.applyColorMap(
        np.uint8(cv2.resize(np.clip(inv_truth / scale, 0, 1), (320, 320),
                            interpolation=cv2.INTER_NEAREST) * 255), cv2.COLORMAP_TURBO,
    )
    prediction_map = cv2.applyColorMap(
        np.uint8(cv2.resize(np.clip(inv_prediction / scale, 0, 1), (320, 320),
                            interpolation=cv2.INTER_NEAREST) * 255), cv2.COLORMAP_TURBO,
    )
    classification = base.copy()
    truth_up = cv2.resize(truth_critical.astype(np.uint8), (320, 320),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    prediction_up = cv2.resize(predicted_critical.astype(np.uint8), (320, 320),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    classification[truth_up & prediction_up] = (0, 220, 0)
    classification[truth_up & ~prediction_up] = (0, 0, 255)
    classification[~truth_up & prediction_up] = (0, 220, 255)
    panels = np.hstack((
        labeled(base, "current image"),
        labeled(truth_map, "truth inverse TTC (bright = urgent)"),
        labeled(prediction_map, "predicted inverse TTC"),
        labeled(classification, "green TP | red miss | yellow false alarm"),
    ))
    banner = np.zeros((46, panels.shape[1], 3), np.uint8)
    cv2.putText(banner, name.replace("_", " "), (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        banner,
        f"threshold {threshold:.4f} | precision {candidate['precision']:.2f} | "
        f"recall {candidate['recall']:.2f} | TP/FP/FN "
        f"{candidate['tp']}/{candidate['fp']}/{candidate['fn']}",
        (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    panel = np.vstack((banner, panels))
    filename = f"{name}.png"
    cv2.imwrite(str(output_dir / filename), panel)
    return {**candidate, "category": name, "file": filename}, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.552)
    parser.add_argument("--threshold-json", type=Path)
    parser.add_argument("--precision-target", default="0.65")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.threshold_json:
        calibration = json.loads(args.threshold_json.read_text())
        args.threshold = calibration["validation"]["precision_constrained"][
            args.precision_target
        ]["threshold"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("critical rendering requires a Slurm GPU allocation")
    model, initialization = load_dory_checkpoint(args.checkpoint, device)
    model.eval()
    dataset = TTCGateDataset(args.dataset, "test")
    candidates = scan(model, dataset, device, args.batch_size, args.workers, args.threshold)
    order = ("true_positive", "dense_critical", "critical_miss", "safe_correct", "false_alarm")
    records, panels = [], []
    for name in order:
        if name in candidates:
            record, panel = render(
                model, dataset, name, candidates[name], device, args.threshold, args.output_dir
            )
            records.append(record)
            panels.append(panel)
    if panels:
        cv2.imwrite(str(args.output_dir / "critical_ttc_predictions_montage.png"),
                    np.vstack(panels))
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization": initialization,
        "threshold": args.threshold,
        "examples": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
