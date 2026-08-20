#!/usr/bin/env python
"""NeMO QAT distillation for the ESPNet/DroNet/gate deployment graphs."""

from __future__ import print_function

import argparse
import copy
import json
from pathlib import Path

import nemo
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gap8_perception.nemo_espnet_dronet_gate_export import (
    EncoderNet,
    MiddleSpatialHead,
    NavigationHead,
    PresenceHead,
    load_gate_head,
    load_navigation_head,
    load_presence_head,
    load_spatial_head,
    temporal_pairs,
)
from gap8_perception.nemo_stdc_shared_dory_export import (
    activation_epsilon, canonicalize_batchnorm_affine,
    configure_residual_requantization, load_archive, set_pulp_int8_precision,
)
from gap8_perception.train_nemo_espnet_dory_qat import (
    PairImages,
    configure_adds,
    evenly_sample,
    freeze_batchnorm_statistics,
    representative_pairs,
    shift_terminal_activation,
)


def safe_integer_add(self, *inputs):
    output = inputs[0].clone()
    for value in inputs[1:]:
        output = output + value
    return output


nemo.quant.pact.PACT_IntegerAdd.forward = safe_integer_add


def build(graph, bridge):
    encoder = EncoderNet()
    load_archive(encoder, bridge / "encoder_float_state.npz")
    if graph == "encoder":
        return encoder, None, encoder.input_shape
    if graph == "corner_head":
        model, loader = MiddleSpatialHead(4), load_spatial_head
    elif graph == "gate_head":
        model, loader = MiddleSpatialHead(1), load_gate_head
    elif graph == "presence_head":
        model, loader = PresenceHead(), load_presence_head
    elif graph == "navigation_head":
        model, loader = NavigationHead(), load_navigation_head
    else:
        raise ValueError(graph)
    loader(model, bridge / (graph + "_float_state.npz"))
    return model, encoder.eval(), model.input_shape


def source_tensor(graph, images, encoder):
    if graph == "encoder":
        return images
    with torch.no_grad():
        return encoder(images)


def build_quantized_encoder(encoder, checkpoint_path):
    """Restore the fake-quant encoder used to condition downstream QAT."""
    quantized = nemo.transform.quantize_pact(
        copy.deepcopy(encoder), dummy_input=torch.ones(1, *encoder.input_shape)
    )
    set_pulp_int8_precision(quantized)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if state.get("graph") != "encoder":
        raise RuntimeError("downstream QAT requires an encoder QAT checkpoint")
    quantized.load_state_dict(state["model"], strict=True)
    return quantized.eval()


class DecodedIntegerEncoder(torch.nn.Module):
    """Expose an ID encoder as physical features to a fake-quantized head."""
    def __init__(self, encoder):
        super(DecodedIntegerEncoder, self).__init__()
        self.encoder = encoder
        self.epsilon = activation_epsilon(encoder)

    def forward(self, images):
        return self.encoder(images * 255.0) * self.epsilon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=(
        "encoder", "corner_head", "gate_head", "presence_head", "navigation_head"
    ), required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-images", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--gate-dataset", type=Path)
    parser.add_argument("--gate-split-file", type=Path)
    parser.add_argument("--no-gate-dataset", type=Path)
    parser.add_argument("--paired-no-gate-dataset", type=Path)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    parser.add_argument("--canonicalize-batchnorm", action="store_true")
    parser.add_argument("--condition-on-integer-encoder", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260820)
    np.random.seed(20260820)
    args.output.mkdir(parents=True, exist_ok=True)

    static_sources = []
    if args.gate_dataset is not None:
        split = json.loads(args.gate_split_file.read_text())
        static_sources.append(("synthetic_gate", args.gate_dataset, split["train"]))
    if args.no_gate_dataset is not None:
        static_sources.append((
            "synthetic_no_gate", args.no_gate_dataset,
            ["shard_%09d" % (index * 1000) for index in range(3)],
        ))
    if args.paired_no_gate_dataset is not None:
        static_sources.append((
            "paired_no_gate", args.paired_no_gate_dataset,
            ["shard_%09d" % (index * 1000) for index in range(16)],
        ))
    pairs, source_counts = representative_pairs(
        temporal_pairs(args.dataset, "train"), static_sources,
        args.real_root, args.train_limit,
    )
    calibration = evenly_sample(pairs, min(args.calibration_images, len(pairs)))
    base, encoder, shape = build(args.graph, args.bridge)
    if args.canonicalize_batchnorm:
        canonicalize_batchnorm_affine(base)
        if encoder is not None:
            canonicalize_batchnorm_affine(encoder)
    base.eval()
    quantized_encoder = None
    if encoder is not None:
        encoder_checkpoint = args.output / "encoder_qat.pt"
        if not encoder_checkpoint.is_file():
            raise RuntimeError(
                "train encoder first; downstream QAT conditions on encoder_qat.pt"
            )
        quantized_encoder = build_quantized_encoder(encoder, encoder_checkpoint)
        if args.condition_on_integer_encoder:
            configure_residual_requantization(quantized_encoder, 1)
            quantized_encoder.qd_stage(
                eps_in=1.0 / 255.0, prune_empty_bn=False
            )
            quantized_encoder.id_stage()
            quantized_encoder = DecodedIntegerEncoder(quantized_encoder).eval()
    output_offset = learned_bias = None
    if encoder is not None:
        learned_bias = base.output_proj[1].bias.detach().cpu().numpy().copy()
        output_offset = shift_terminal_activation(
            base, quantized_encoder, calibration, args.batch_size
        )
    teacher = copy.deepcopy(base).eval()
    student = nemo.transform.quantize_pact(base, dummy_input=torch.ones(1, *shape))
    set_pulp_int8_precision(student)
    student.reset_alpha_weights()
    student.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration), args.batch_size):
            batch_pairs = calibration[start:start + args.batch_size]
            images = torch.stack([PairImages(batch_pairs)[i] for i in range(len(batch_pairs))])
            student(source_tensor(
                args.graph, images,
                quantized_encoder if quantized_encoder is not None else encoder,
            ))
    student.unset_statistics_act()
    student.reset_alpha_act()
    configure_adds(student, args.residual_requantization_factor)

    loader = DataLoader(
        PairImages(pairs), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    history, best_loss = [], None
    for epoch in range(1, args.epochs + 1):
        student.train()
        freeze_batchnorm_statistics(student)
        total = examples = 0
        for images in loader:
            with torch.no_grad():
                teacher_source = source_tensor(args.graph, images, encoder)
                student_source = source_tensor(
                    args.graph, images,
                    quantized_encoder if quantized_encoder is not None else encoder,
                )
                target = teacher(teacher_source)
            prediction = student(student_source)
            error = prediction - target
            if args.graph in ("corner_head", "gate_head"):
                channels = prediction.shape[1]
                broadcast = (1, channels) + (1,) * (prediction.ndim - 2)
                offset_tensor = torch.as_tensor(
                    output_offset, dtype=prediction.dtype
                ).view(broadcast)
                bias_tensor = torch.as_tensor(
                    learned_bias, dtype=prediction.dtype
                ).view(broadcast)
                decoded_prediction = prediction - offset_tensor + bias_tensor
                decoded_target = target - offset_tensor + bias_tensor
                loss = F.smooth_l1_loss(prediction, target)
                loss = loss + 5.0 * F.mse_loss(
                    decoded_prediction.sigmoid(), decoded_target.sigmoid()
                )
            else:
                loss = error.pow(2).mean() + 0.02 * error.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * images.shape[0]
            examples += images.shape[0]
        metric = {"epoch": epoch, "distillation_loss": total / examples}
        history.append(metric)
        checkpoint = {
            "graph": args.graph, "model": student.state_dict(), "precision_bits": 8,
            "activation_bits": 8, "signed_weight_bits": 7,
            "canonical_batchnorm_affine": args.canonicalize_batchnorm,
            "conditioned_on_integer_encoder": args.condition_on_integer_encoder,
            "residual_requantization_factor": args.residual_requantization_factor,
            "source_bridge": str(args.bridge),
            "output_offset": output_offset.tolist() if output_offset is not None else None,
            "learned_bias": learned_bias.tolist() if learned_bias is not None else None,
            "representative_source_counts": source_counts,
            "calibration_examples": len(calibration), "training_examples": len(pairs),
            "epoch": epoch, "distillation_loss": metric["distillation_loss"],
        }
        torch.save(checkpoint, str(args.output / (args.graph + "_qat_last.pt")),
                   _use_new_zipfile_serialization=False)
        if best_loss is None or metric["distillation_loss"] < best_loss:
            best_loss = metric["distillation_loss"]
            torch.save(checkpoint, str(args.output / (args.graph + "_qat.pt")),
                       _use_new_zipfile_serialization=False)
        print(json.dumps(metric), flush=True)
    (args.output / (args.graph + "_qat_history.json")).write_text(
        json.dumps(history, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
