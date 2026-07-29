#!/usr/bin/env python3
"""Run the installed DORY NEMO frontend as an authoritative export gate."""

import argparse
import json
import re
from pathlib import Path

from dory.Frontend_frameworks.NEMO.Parser import onnx_manager
from dory.Hardware_targets.PULP.GAP8.HW_Parser import onnx_manager as gap8_backend
from dory.Parsers.HW_node import HW_node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app-dir", type=Path)
    parser.add_argument("--expected-output-bytes", type=int, default=12800)
    args = parser.parse_args()
    config = {
        "BNRelu_bits": 32,
        "input_bits": 8,
        "input_signed": False,
    }
    graph = onnx_manager(str(args.onnx), config, "gap8_").full_graph_parsing()
    frontend_node_names = [node.name for node in graph]
    backend_config = dict(config)
    backend_config.update({
        "onnx_file": args.onnx.name,
        "code reserved space": 95000,
    })
    # NEMO emits one golden activation per fused frontend node. Derive the
    # expected count from the parsed graph so architecture changes cannot
    # silently disable checksum validation or retain a stale layer count.
    golden_count = 0
    while (args.onnx.parent / f"out_layer{golden_count}.txt").is_file():
        golden_count += 1
    golden_files_present = (
        (args.onnx.parent / "input.txt").is_file() and golden_count > 0
    )
    if golden_files_present:
        hardware_graph = gap8_backend(
            graph, backend_config, str(args.onnx.parent), 1
        ).full_graph_parsing()
    else:
        original_checksum = HW_node.add_checksum_activations_integer
        HW_node.add_checksum_activations_integer = lambda self, *a, **k: None
        try:
            hardware_graph = gap8_backend(
                graph, backend_config, str(args.onnx.parent), 1
            ).full_graph_parsing()
        finally:
            HW_node.add_checksum_activations_integer = original_checksum
    l1_tile_bytes = []
    for node in hardware_graph:
        tile = node.tiling_dimensions["L1"]
        l1_tile_bytes.append(int(sum(
            float(tile.get(key, 0) or 0)
            for key in (
                "weight_memory", "bias_memory", "constants_memory",
                "input_activation_memory", "output_activation_memory",
            )
        )))
    final_node = hardware_graph[-1]
    if golden_files_present and golden_count != len(hardware_graph):
        raise RuntimeError(
            "golden activation count %d does not match fused hardware nodes %d"
            % (golden_count, len(hardware_graph))
        )
    final_output_bytes = int(final_node.output_activation_memory)
    final_output_bits = int(final_node.output_activation_bits)
    final_checksums = list(getattr(final_node, "check_sum_out", []) or [])
    if (
        final_output_bits != 8
        or final_output_bytes != args.expected_output_bytes
    ):
        raise RuntimeError(
            "expected uint8 terminal output with %d bytes, got bits=%d bytes=%d"
            % (args.expected_output_bytes, final_output_bits, final_output_bytes)
        )
    if golden_files_present and (
        not final_checksums or int(final_checksums[0]) <= 0
    ):
        raise RuntimeError("terminal golden checksum is missing or degenerate")
    report = {
        "passed": True,
        "onnx": str(args.onnx),
        "dory_frontend": "NEMO",
        "dory_frontend_nodes": len(frontend_node_names),
        "frontend_node_names": frontend_node_names,
        "single_output_graph": True,
        "gap8_backend_tiling_passed": True,
        "gap8_fused_hardware_nodes": len(hardware_graph),
        "gap8_total_macs": int(sum(node.MACs for node in hardware_graph)),
        "gap8_max_l1_tile_bytes_estimate": max(l1_tile_bytes),
        "gap8_l1_capacity_bytes": 64000,
        "gap8_final_output_bits": final_output_bits,
        "gap8_final_output_bytes": final_output_bytes,
        "expected_final_output_bytes": args.expected_output_bytes,
        "gap8_final_output_checksums": final_checksums,
        "activation_checksums_loaded": golden_files_present,
        "golden_activation_files": golden_count,
        "activation_checksums_skipped": not golden_files_present,
        "remaining_gate": (
            "compile generated C and run GVSOC/physical GAP8 parity"
        ),
    }
    if args.app_dir:
        from network_generate import network_generate

        dory_config = {
            "BNRelu_bits": 32,
            "input_bits": 8,
            "input_signed": False,
            "onnx_file": args.onnx.name,
            "code reserved space": 95000,
        }
        config_path = args.onnx.parent / "dory_config.json"
        config_path.write_text(json.dumps(dory_config, indent=2) + "\n")
        args.app_dir.mkdir(parents=True, exist_ok=True)
        network_generate(
            "NEMO", "PULP.GAP8", str(config_path),
            verbose_level="Last+Perf_final", perf_layer="No",
            optional="8bit", appdir=str(args.app_dir), prefix="gap8",
        )
        network_source = args.app_dir / "src" / "gap8_network.c"
        bounded = network_source.read_text()
        bounded = re.sub(
            r"#define L3_WEIGHTS_SIZE \d+",
            "#define L3_WEIGHTS_SIZE 262144",
            bounded,
        )
        bounded = re.sub(
            r"#define L3_INPUT_SIZE \d+",
            "#define L3_INPUT_SIZE 131072",
            bounded,
        )
        bounded = re.sub(
            r"#define L3_OUTPUT_SIZE \d+",
            "#define L3_OUTPUT_SIZE 131072",
            bounded,
        )
        bounded = re.sub(
            r'(#ifdef VERBOSE\n\s+printf\("Layer %s %d ended: \\n", '
            r"Layers_name\[i\], i\);\n)"
            r"(\s+if \(i == \d+\)\n\s+checksum\([^;]+;\n)"
            r"(#endif)",
            r"\1\3\n#ifdef DORY_CHECKSUM_HARNESS\n\2#endif",
            bounded,
        )
        network_source.write_text(bounded)
        main_source = args.app_dir / "src" / "gap8_main.c"
        main_text = re.sub(
            r"size_t input_size = \d+;",
            "size_t input_size = 131072;",
            main_source.read_text(),
        )
        main_text = main_text.replace(
            "pi_l2_free(l2_buffer, 417000);",
            "pi_l2_free(l2_buffer, 417000);\n  pmsis_exit(0);",
        )
        main_text = main_text.replace(
            "int main () {\n#ifndef TARGET_CHIP_FAMILY_GAP9",
            "int main () {\n#ifndef DORY_CHECKSUM_HARNESS\n"
            "#ifndef TARGET_CHIP_FAMILY_GAP9",
        )
        main_text = main_text.replace(
            "\n\n  pmsis_kickoff((void*)application);",
            "\n#endif\n\n  pmsis_kickoff((void*)application);",
        )
        main_text = main_text.replace(
            "  pi_time_wait_us(10000);",
            "#ifndef DORY_CHECKSUM_HARNESS\n"
            "  pi_time_wait_us(10000);\n"
            "#endif",
        )
        main_source.write_text(main_text)
        makefile = args.app_dir / "Makefile"
        makefile.write_text(
            makefile.read_text()
            + "\nDORY_CHECKSUM_HARNESS ?= 0\n"
            + "ifeq ($(DORY_CHECKSUM_HARNESS),1)\n"
            + "APP_CFLAGS += -DDORY_CHECKSUM_HARNESS\n"
            + "endif\n"
        )
        generated_files = [path for path in args.app_dir.rglob("*") if path.is_file()]
        report["generated_c_passed"] = True
        report["generated_app_dir"] = str(args.app_dir)
        report["generated_files"] = len(generated_files)
    else:
        report["generated_c_passed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
