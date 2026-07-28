#!/usr/bin/env python3
"""Write input/target/prediction panels for all three perception heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from gap8_perception.data import MultiTaskDataset
from gap8_perception.danger_postprocessor import collision_probability_from_range
from gap8_perception.evaluate import local_centroid
from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.quantization import prepare_int8_qat


def label(panel: np.ndarray, text: str) -> np.ndarray:
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (160, 18), (0, 0, 0), -1)
    cv2.putText(
        output, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Gap8MultiTaskNet(state.get("gate_head", True), state.get("state_dim", 8))
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = MultiTaskDataset(
        args.dataset, args.targets, args.split_file, args.split, limit=args.count
    )
    colors = np.asarray([[255, 0, 0], [0, 255, 255], [255, 0, 255], [0, 165, 255]])
    with torch.no_grad():
        for index in range(min(args.count, len(dataset))):
            batch = dataset[index]
            image = batch["image"].unsqueeze(0).to(device)
            vehicle_state = batch["vehicle_state"].unsqueeze(0).to(device)
            output = model(image, vehicle_state)
            mono = np.repeat((batch["image"][0].numpy() * 255).astype(np.uint8)[..., None], 3, 2)
            corner_panel = mono.copy()
            truth_xy = batch["corner_xy"].numpy()
            if batch["corner_valid"]:
                for point, color in zip(truth_xy, colors):
                    cv2.circle(corner_panel, tuple(np.rint(point).astype(int)), 3, color.tolist(), -1)
            corner_probability = output["corners"].sigmoid()
            pred_xy = local_centroid(corner_probability).cpu().numpy()[0] * 4
            confidence = corner_probability.flatten(2).amax(2).cpu().numpy()[0]
            for point, color, score in zip(pred_xy, colors, confidence):
                if score >= 0.20:
                    cv2.drawMarker(
                        corner_panel,
                        tuple(np.rint(point).astype(int)),
                        color.tolist(),
                        cv2.MARKER_CROSS,
                        7,
                        1,
                    )
            danger_t = cv2.resize(batch["danger"][0].numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            nominal_danger_p20 = output["danger"].sigmoid()[0, 0].cpu().numpy()
            inverse_range_t = cv2.resize(batch["urgency"][0].numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            inverse_range_p = cv2.resize(output["urgency"].sigmoid()[0, 0].cpu().numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            uncertainty_t = cv2.resize(batch["uncertainty"][0].numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            uncertainty_p = cv2.resize(output["uncertainty"].sigmoid()[0, 0].cpu().numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            state_values = batch["vehicle_state"].numpy()
            final_danger_p20, _ = collision_probability_from_range(
                output["urgency"].sigmoid()[0, 0].cpu().numpy(),
                nominal_danger_p20,
                output["uncertainty"].sigmoid()[0, 0].cpu().numpy(),
                body_speed_mps=float(state_values[0]),
                horizon_s=float(state_values[6]),
                latency_s=float(state_values[7]),
            )
            nominal_danger_p = cv2.resize(
                nominal_danger_p20, (160, 160), interpolation=cv2.INTER_NEAREST
            )
            danger_p = cv2.resize(
                final_danger_p20, (160, 160), interpolation=cv2.INTER_NEAREST
            )
            gate_t = cv2.resize(batch["gate"][0].numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
            panels = [
                label(mono, "HM01B0"),
                label(corner_panel, "corners GT dots / pred +"),
                label(cv2.applyColorMap(np.uint8(danger_t * 255), cv2.COLORMAP_TURBO), "collision GT 1m/s"),
                label(cv2.applyColorMap(np.uint8(nominal_danger_p * 255), cv2.COLORMAP_TURBO), "nominal collision pred"),
                label(cv2.applyColorMap(np.uint8(danger_p * 255), cv2.COLORMAP_TURBO), "controller danger pred"),
                label(cv2.applyColorMap(np.uint8(inverse_range_t * 255), cv2.COLORMAP_TURBO), "inverse range GT"),
                label(cv2.applyColorMap(np.uint8(inverse_range_p * 255), cv2.COLORMAP_TURBO), "inverse range pred"),
                label(cv2.applyColorMap(np.uint8(uncertainty_t * 255), cv2.COLORMAP_MAGMA), "uncertainty GT"),
                label(cv2.applyColorMap(np.uint8(uncertainty_p * 255), cv2.COLORMAP_MAGMA), "uncertainty pred"),
                label(cv2.applyColorMap(np.uint8(gate_t * 255), cv2.COLORMAP_VIRIDIS), "gate opening GT"),
            ]
            if "gate" in output:
                gate_p = cv2.resize(output["gate"].sigmoid()[0, 0].cpu().numpy(), (160, 160), interpolation=cv2.INTER_NEAREST)
                panels.append(
                    label(
                        cv2.applyColorMap(
                            np.uint8(gate_p * 255), cv2.COLORMAP_VIRIDIS
                        ),
                        "gate opening pred",
                    )
                )
            cv2.imwrite(str(args.output / f"prediction_{index:04d}.png"), np.hstack(panels))


if __name__ == "__main__":
    main()
