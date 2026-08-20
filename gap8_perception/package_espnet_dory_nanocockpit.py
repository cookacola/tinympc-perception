#!/usr/bin/env python3
"""Package four verified DORY graphs as a temporal NanoCockpit network."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from gap8_perception.package_stdc_pair_nanocockpit import (
    COMMON_SOURCES,
    copy_namespaced_graph,
    normalized_generated_text,
)


GRAPH_BYTES = {
    "encoder": 32 * 40 * 40,
    "corner_head": 4 * 40 * 40,
    "gate_head": 40 * 40,
    "danger_head": 10 * 10,
}


def quantized_threshold(graph, probability):
    logit = math.log(probability / (1.0 - probability))
    value = (
        logit + graph["output_offset"][0] - graph["learned_bias"][0]
    ) / graph["output_epsilon"]
    return max(0, min(255, int(math.floor(value)) + 1))


def main():
    parser = argparse.ArgumentParser()
    for name in ("encoder", "corner-head", "gate-head", "danger-head"):
        parser.add_argument(f"--{name}-app", type=Path, required=True)
    parser.add_argument("--nemo-report", type=Path, required=True)
    parser.add_argument("--integer-evaluation", type=Path, required=True)
    parser.add_argument("--student-summary", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--name", default="gap8-espnet-dory-student-v1")
    args = parser.parse_args()
    destination = args.destination / args.name
    if destination.exists():
        raise FileExistsError(destination)
    for directory in ("src", "inc", "hex"):
        (destination / directory).mkdir(parents=True)

    apps = {
        "encoder": args.encoder_app,
        "corner_head": args.corner_head_app,
        "gate_head": args.gate_head_app,
        "danger_head": args.danger_head_app,
    }
    common_app = args.encoder_app
    for filename in COMMON_SOURCES:
        source = common_app / "src" / filename
        (destination / "src" / filename).write_text(
            normalized_generated_text(source.read_text())
        )
    for path in sorted((common_app / "inc").glob("*.h")):
        if not path.name.startswith("gap8_"):
            (destination / "inc" / path.name).write_text(
                normalized_generated_text(path.read_text())
            )
    for graph, app in apps.items():
        copy_namespaced_graph(app, destination, graph)

    (destination / "inc" / "network.h").write_text("""#ifndef ESPNET_DORY_NETWORK_H
#define ESPNET_DORY_NETWORK_H
#include <stddef.h>
#include "pmsis.h"
void network_initialize(void);
void network_terminate(void);
void network_run_async_cl(void *l2_buffer, size_t l2_buffer_size,
                          void *l2_final_output, int exec, int initial_dir,
                          pi_device_t *cluster, pi_task_t *network_done);
#endif
""")
    (destination / "src" / "espnet_dory_network.c").write_text("""#include "network.h"
#include "stdc_encoder_network.h"
#include "stdc_corner_head_network.h"
#include "stdc_gate_head_network.h"
#include "stdc_danger_head_network.h"
#include <string.h>

#define SHARED_BYTES 51200
#define CORNER_BYTES 6400
#define GATE_BYTES 1600
#define DANGER_BYTES 100

static PI_L2 uint8_t shared_output[SHARED_BYTES];
static PI_L2 uint8_t corner_output[CORNER_BYTES];
static PI_L2 uint8_t gate_output[GATE_BYTES];
static PI_L2 uint8_t danger_output[DANGER_BYTES];
static PI_FC_L1 pi_task_t encoder_done, corner_done, gate_done, danger_done;
static void *workspace, *combined_output;
static size_t workspace_size;
static int run_exec, run_initial_dir;
static pi_device_t *run_cluster;
static pi_task_t *user_done;

static void danger_finished(void *arg) {
  (void)arg;
  memcpy(combined_output, corner_output, CORNER_BYTES);
  memcpy((uint8_t *)combined_output + CORNER_BYTES, gate_output, GATE_BYTES);
  memcpy((uint8_t *)combined_output + CORNER_BYTES + GATE_BYTES,
         danger_output, DANGER_BYTES);
  pi_task_push(user_done);
}
static void gate_finished(void *arg) {
  (void)arg;
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_danger_head_network_run_async_cl(
      workspace, workspace_size, danger_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&danger_done, danger_finished, NULL));
}
static void corner_finished(void *arg) {
  (void)arg;
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_gate_head_network_run_async_cl(
      workspace, workspace_size, gate_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&gate_done, gate_finished, NULL));
}
static void encoder_finished(void *arg) {
  (void)arg;
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_corner_head_network_run_async_cl(
      workspace, workspace_size, corner_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&corner_done, corner_finished, NULL));
}
void network_initialize(void) {
  stdc_encoder_network_initialize();
  stdc_corner_head_network_initialize();
  stdc_gate_head_network_initialize();
  stdc_danger_head_network_initialize();
}
void network_terminate(void) {
  stdc_encoder_network_terminate();
  stdc_corner_head_network_terminate();
  stdc_gate_head_network_terminate();
  stdc_danger_head_network_terminate();
}
void network_run_async_cl(void *l2_buffer, size_t l2_buffer_size,
                          void *l2_final_output, int exec, int initial_dir,
                          pi_device_t *cluster, pi_task_t *network_done) {
  workspace = l2_buffer;
  workspace_size = l2_buffer_size;
  combined_output = l2_final_output;
  run_exec = exec;
  run_initial_dir = initial_dir;
  run_cluster = cluster;
  user_done = network_done;
  stdc_encoder_network_run_async_cl(
      workspace, workspace_size, shared_output, exec, initial_dir, cluster,
      pi_task_callback(&encoder_done, encoder_finished, NULL));
}
""")

    nemo = json.loads(args.nemo_report.read_text())
    graphs = {item["graph"]: item for item in nemo["graphs"]}
    if set(graphs) != set(GRAPH_BYTES):
        raise RuntimeError("unexpected NEMO graph set")
    evaluation = json.loads(args.integer_evaluation.read_text())
    danger_probability = float(evaluation["selected_danger_threshold"])
    corner_q = []
    for channel in range(4):
        graph = dict(graphs["corner_head"])
        graph["output_offset"] = [graphs["corner_head"]["output_offset"][channel]]
        graph["learned_bias"] = [graphs["corner_head"]["learned_bias"][channel]]
        corner_q.append(quantized_threshold(graph, 0.25))
    gate_q = quantized_threshold(graphs["gate_head"], 0.5)
    danger_q = quantized_threshold(graphs["danger_head"], danger_probability)
    header = """#ifndef GAP8_PERCEPTION_OUTPUT_H
#define GAP8_PERCEPTION_OUTPUT_H
#include <stdint.h>
#define GAP8_INPUT_WIDTH 160
#define GAP8_INPUT_HEIGHT 160
#define GAP8_INPUT_CHANNELS 2
#define GAP8_OUTPUT_BYTES 8100
#define GAP8_CONTROL_WIDTH 20
#define GAP8_CONTROL_HEIGHT 20
#define GAP8_CORNER_Q_THRESHOLD_0 %d
#define GAP8_CORNER_Q_THRESHOLD_1 %d
#define GAP8_CORNER_Q_THRESHOLD_2 %d
#define GAP8_CORNER_Q_THRESHOLD_3 %d
#define GAP8_GATE_Q_THRESHOLD %d
#define GAP8_DANGER_Q_THRESHOLD %d
#define GAP8_DANGER_QUANT_EPSILON %.10gf
#define GAP8_DANGER_QUANT_OFFSET %.10gf
#define GAP8_DANGER_QUANT_BIAS %.10gf
#define GAP8_DANGER_PROBABILITY_THRESHOLD %.10gf
void gap8_decode_corner_argmax(const uint8_t *packed,
                               float corners_xy[8], uint8_t confidence[4]);
int gap8_validate_or_recover_gate(float corners_xy[8],
                                  const uint8_t confidence[4]);
void gap8_pool_control_maps(const uint8_t *packed,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]);
#endif
""" % (
        *corner_q, gate_q, danger_q,
        graphs["danger_head"]["output_epsilon"],
        graphs["danger_head"]["output_offset"][0],
        graphs["danger_head"]["learned_bias"][0],
        danger_probability,
    )
    (destination / "inc" / "gap8_perception_output.h").write_text(header)
    postprocess = Path(__file__).parent / "firmware" / "espnet_dory_output.c"
    shutil.copy2(postprocess, destination / "src" / "gap8_perception_output.c")

    weights = sorted(path.name for path in (destination / "hex").glob("*_weights.hex"))
    network_mk = """# Generated two-frame ESPNet DORY student package.
CORE ?= 8
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM
APP_SRCS += $(wildcard $(NETWORK_DIR)/src/*.c)
APP_CFLAGS += -I$(NETWORK_DIR)/inc
APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP8_MULTITASK_NETWORK=1
APP_CFLAGS += -DGAP8_TEMPORAL_NETWORK=1 -DGAP8_ESPNET_DORY_STUDENT=1
APP_CFLAGS += -Wno-error -O2 -fno-indirect-inlining -flto
APP_LDFLAGS += -lm -flto
APP_CFLAGS += -DGAP_SDK=1 -DFLASH_TYPE=$(FLASH_TYPE)
APP_CFLAGS += -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS -DFS_READ_FS
"""
    for filename in weights:
        network_mk += "FLASH_FILES += $(NETWORK_DIR)/hex/%s\n" % filename
    network_mk += "READFS_FILES += $(FLASH_FILES)\n"
    (destination / "network.mk").write_text(network_mk)

    student = json.loads(args.student_summary.read_text())
    manifest = {
        "format": "nanocockpit-espnet-dory-student-v1",
        "name": args.name,
        "input": {"dtype": "uint8", "layout": "HWC", "shape": [160, 160, 2],
                  "channel_order": ["previous", "current"]},
        "outputs": {
            "corner": {"shape": [40, 40, 4], "bytes": 6400},
            "gate": {"shape": [40, 40, 1], "bytes": 1600},
            "danger": {"shape": [10, 10, 1], "bytes": 100},
        },
        "integer_affine": {
            name: {key: graphs[name + "_head"][key]
                   for key in ("output_epsilon", "output_offset", "learned_bias")}
            for name in ("corner", "gate", "danger")
        },
        "thresholds": {
            "danger_probability": danger_probability,
            "danger_uint8": danger_q,
            "gate_probability": 0.5,
            "gate_uint8": gate_q,
            "corner_uint8": corner_q,
        },
        "student_float_test": {
            "obstacle": student["obstacle_test"],
            "gate": student["gate_test"],
        },
        "integer_test": evaluation,
        "source_reports": {
            "nemo": str(args.nemo_report.resolve()),
            "integer_evaluation": str(args.integer_evaluation.resolve()),
            "student_summary": str(args.student_summary.resolve()),
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
