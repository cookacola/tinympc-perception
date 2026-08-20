#!/usr/bin/env python3
"""Package verified ESPNet/DroNet/gate DORY graphs for NanoCockpit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gap8_perception.package_stdc_pair_nanocockpit import (
    COMMON_SOURCES,
    copy_namespaced_graph,
    normalized_generated_text,
)


GRAPHS = ("encoder", "corner_head", "gate_head", "presence_head", "navigation_head")


def c_literal(value):
    text = "%.10g" % float(value)
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def c_array(values):
    return ", ".join(c_literal(value) for value in values)


def main():
    parser = argparse.ArgumentParser()
    for graph in GRAPHS:
        parser.add_argument("--" + graph.replace("_", "-") + "-app", type=Path, required=True)
    parser.add_argument("--nemo-report", type=Path, required=True)
    parser.add_argument("--integer-evaluation", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--navigation-calibration", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--name", default="gap8-espnet-dronet-gate-v1")
    args = parser.parse_args()
    destination = args.destination / args.name
    if destination.exists():
        raise FileExistsError(destination)
    for directory in ("src", "inc", "hex"):
        (destination / directory).mkdir(parents=True)
    apps = {
        graph: getattr(args, graph + "_app") for graph in GRAPHS
    }
    for filename in COMMON_SOURCES:
        source = apps["encoder"] / "src" / filename
        (destination / "src" / filename).write_text(
            normalized_generated_text(source.read_text())
        )
    for path in sorted((apps["encoder"] / "inc").glob("*.h")):
        if not path.name.startswith("gap8_"):
            (destination / "inc" / path.name).write_text(
                normalized_generated_text(path.read_text())
            )
    for graph, app in apps.items():
        copy_namespaced_graph(app, destination, graph)

    (destination / "inc" / "network.h").write_text("""#ifndef ESPNET_DRONET_GATE_NETWORK_H
#define ESPNET_DRONET_GATE_NETWORK_H
#include <stddef.h>
#include "pmsis.h"
void network_initialize(void);
void network_terminate(void);
void network_run_async_cl(void *l2_buffer, size_t l2_buffer_size,
                          void *l2_final_output, int exec, int initial_dir,
                          pi_device_t *cluster, pi_task_t *network_done);
#endif
""")
    (destination / "src" / "espnet_dronet_gate_network.c").write_text("""#include "network.h"
#include "stdc_encoder_network.h"
#include "stdc_corner_head_network.h"
#include "stdc_gate_head_network.h"
#include "stdc_presence_head_network.h"
#include "stdc_navigation_head_network.h"
#include <string.h>

#define SHARED_BYTES 25600
#define CORNER_BYTES 1600
#define GATE_BYTES 400
#define PRESENCE_BYTES 1
#define NAVIGATION_BYTES 2

static PI_L2 uint8_t shared_output[SHARED_BYTES];
static PI_L2 uint8_t corner_output[CORNER_BYTES];
static PI_L2 uint8_t gate_output[GATE_BYTES];
static PI_L2 uint8_t presence_output[PRESENCE_BYTES];
static PI_L2 uint8_t navigation_output[NAVIGATION_BYTES];
static PI_FC_L1 pi_task_t encoder_done, corner_done, gate_done, presence_done, navigation_done;
static void *workspace, *combined_output;
static size_t workspace_size;
static int run_exec, run_initial_dir;
static pi_device_t *run_cluster;
static pi_task_t *user_done;

static void navigation_finished(void *arg) {
  (void)arg;
#ifdef TINY_RACER_PARITY_TEST
  printf("PARITY stage=navigation\\n");
#endif
  memcpy(combined_output, corner_output, CORNER_BYTES);
  memcpy((uint8_t *)combined_output + CORNER_BYTES, gate_output, GATE_BYTES);
  memcpy((uint8_t *)combined_output + CORNER_BYTES + GATE_BYTES,
         presence_output, PRESENCE_BYTES);
  memcpy((uint8_t *)combined_output + CORNER_BYTES + GATE_BYTES + PRESENCE_BYTES,
         navigation_output, NAVIGATION_BYTES);
  pi_task_push(user_done);
}
static void presence_finished(void *arg) {
  (void)arg;
#ifdef TINY_RACER_PARITY_TEST
  printf("PARITY stage=presence\\n");
#endif
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_navigation_head_network_run_async_cl(
      workspace, workspace_size, navigation_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&navigation_done, navigation_finished, NULL));
}
static void gate_finished(void *arg) {
  (void)arg;
#ifdef TINY_RACER_PARITY_TEST
  printf("PARITY stage=gate\\n");
#endif
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_presence_head_network_run_async_cl(
      workspace, workspace_size, presence_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&presence_done, presence_finished, NULL));
}
static void corner_finished(void *arg) {
  (void)arg;
#ifdef TINY_RACER_PARITY_TEST
  printf("PARITY stage=corner\\n");
#endif
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_gate_head_network_run_async_cl(
      workspace, workspace_size, gate_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&gate_done, gate_finished, NULL));
}
static void encoder_finished(void *arg) {
  (void)arg;
#ifdef TINY_RACER_PARITY_TEST
  printf("PARITY stage=encoder\\n");
#endif
  memcpy(workspace, shared_output, SHARED_BYTES);
  stdc_corner_head_network_run_async_cl(
      workspace, workspace_size, corner_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&corner_done, corner_finished, NULL));
}
void network_initialize(void) {
  stdc_encoder_network_initialize();
  stdc_corner_head_network_initialize();
  stdc_gate_head_network_initialize();
  stdc_presence_head_network_initialize();
  stdc_navigation_head_network_initialize();
}
void network_terminate(void) {
  stdc_encoder_network_terminate();
  stdc_corner_head_network_terminate();
  stdc_gate_head_network_terminate();
  stdc_presence_head_network_terminate();
  stdc_navigation_head_network_terminate();
}
void network_run_async_cl(void *l2_buffer, size_t l2_buffer_size,
                          void *l2_final_output, int exec, int initial_dir,
                          pi_device_t *cluster, pi_task_t *network_done) {
  workspace = l2_buffer; workspace_size = l2_buffer_size;
  combined_output = l2_final_output; run_exec = exec;
  run_initial_dir = initial_dir; run_cluster = cluster; user_done = network_done;
  stdc_encoder_network_run_async_cl(
      workspace, workspace_size, shared_output, exec, initial_dir, cluster,
      pi_task_callback(&encoder_done, encoder_finished, NULL));
}
""")

    nemo = json.loads(args.nemo_report.read_text())
    graphs = {item["graph"]: item for item in nemo["graphs"]}
    if set(graphs) != set(GRAPHS):
        raise RuntimeError("unexpected NEMO graph set")
    scales = nemo["deployment_logit_scale"]
    training = json.loads(args.training_summary.read_text())
    navigation_calibration = json.loads(args.navigation_calibration.read_text())
    fusion = training["structured_confidence_fusion"]
    corner_thresholds = [
        int(round((offset - bias) / graphs["corner_head"]["output_epsilon"]))
        for offset, bias in zip(graphs["corner_head"]["output_offset"],
                                graphs["corner_head"]["learned_bias"])
    ]
    corner_thresholds = [max(0, min(255, value)) for value in corner_thresholds]
    header = """#ifndef GAP8_PERCEPTION_OUTPUT_H
#define GAP8_PERCEPTION_OUTPUT_H
#include <math.h>
#include <stdint.h>
#define GAP8_INPUT_WIDTH 160
#define GAP8_INPUT_HEIGHT 160
#define GAP8_INPUT_CHANNELS 2
#define GAP8_OUTPUT_BYTES 2003
#define GAP8_CORNER_OFFSET 0
#define GAP8_GATE_OFFSET 1600
#define GAP8_PRESENCE_OFFSET 2000
#define GAP8_NAVIGATION_OFFSET 2001
static const float gap8_corner_epsilon = %.10gf;
static const float gap8_gate_epsilon = %.10gf;
static const float gap8_presence_epsilon = %.10gf;
static const float gap8_navigation_epsilon = %.10gf;
static const float gap8_corner_offset[4] = {%s};
static const float gap8_corner_bias[4] = {%s};
static const float gap8_gate_offset = %.10gf;
static const float gap8_gate_bias = %.10gf;
static const float gap8_gate_logit_scale = %s;
static const float gap8_presence_offset = %.10gf;
static const float gap8_presence_bias = %.10gf;
static const float gap8_navigation_offset[2] = {%s};
static const float gap8_navigation_bias[2] = {%s};
static const float gap8_presence_logit_scale = %s;
static const float gap8_navigation_logit_scale[2] = {%s};
static const float gap8_navigation_yaw_calibration[2] = {%s};
static const float gap8_navigation_collision_calibration[2] = {%s};
static const float gap8_confidence_mean[3] = {%s};
static const float gap8_confidence_scale[3] = {%s};
static const float gap8_confidence_weight[3] = {%s};
static const float gap8_confidence_bias = %.10gf;
static const float gap8_confidence_threshold = %.10gf;
static const float gap8_collision_probability_threshold = %.10gf;
static const uint8_t gap8_corner_q_threshold[4] = {%d, %d, %d, %d};
static inline void gap8_decode_corner_argmax(const uint8_t *packed,
                                              float corners_xy[8],
                                              uint8_t confidence[4]) {
  for (int channel = 0; channel < 4; ++channel) {
    uint8_t best = 0; int best_x = 0, best_y = 0;
    for (int y = 0; y < 20; ++y) for (int x = 0; x < 20; ++x) {
      const uint8_t value = packed[(y * 20 + x) * 4 + channel];
      if (value > best) { best = value; best_x = x; best_y = y; }
    }
    corners_xy[2 * channel] = (best_x + 0.5f) * 8.0f;
    corners_xy[2 * channel + 1] = (best_y + 0.5f) * 8.0f;
    confidence[channel] = best;
  }
}
static inline int gap8_validate_or_recover_gate(float corners_xy[8],
                                                 const uint8_t confidence[4]) {
  (void)corners_xy; int valid = 0;
  for (int i = 0; i < 4; ++i) valid += confidence[i] >= gap8_corner_q_threshold[i];
  return valid >= 3 ? 4 : -1;
}
static inline float gap8_decode_presence_logit(uint8_t quantized) {
  return (((float)quantized * gap8_presence_epsilon
           - gap8_presence_offset + gap8_presence_bias)
          * gap8_presence_logit_scale);
}
static inline void gap8_decode_navigation(const uint8_t quantized[2],
                                           float *yaw,
                                           float *collision_probability) {
  float decoded_yaw =
      ((float)quantized[0] * gap8_navigation_epsilon
       - gap8_navigation_offset[0] + gap8_navigation_bias[0])
      * gap8_navigation_logit_scale[0];
  float collision_logit =
      ((float)quantized[1] * gap8_navigation_epsilon
       - gap8_navigation_offset[1] + gap8_navigation_bias[1])
      * gap8_navigation_logit_scale[1];
  decoded_yaw = decoded_yaw * gap8_navigation_yaw_calibration[0]
                + gap8_navigation_yaw_calibration[1];
  collision_logit =
      collision_logit * gap8_navigation_collision_calibration[0]
      + gap8_navigation_collision_calibration[1];
  *yaw = decoded_yaw;
  *collision_probability = 1.0f / (1.0f + expf(-collision_logit));
}
#endif
""" % (
        graphs["corner_head"]["output_epsilon"],
        graphs["gate_head"]["output_epsilon"],
        graphs["presence_head"]["output_epsilon"],
        graphs["navigation_head"]["output_epsilon"],
        c_array(graphs["corner_head"]["output_offset"]),
        c_array(graphs["corner_head"]["learned_bias"]),
        graphs["gate_head"]["output_offset"][0],
        graphs["gate_head"]["learned_bias"][0],
        c_literal(scales["gate_head"][0]),
        graphs["presence_head"]["output_offset"][0],
        graphs["presence_head"]["learned_bias"][0],
        c_array(graphs["navigation_head"]["output_offset"]),
        c_array(graphs["navigation_head"]["learned_bias"]),
        c_literal(scales["presence_head"][0]), c_array(scales["navigation_head"]),
        c_array((navigation_calibration["yaw_scale"],
                 navigation_calibration["yaw_bias"])),
        c_array((navigation_calibration["collision_logit_scale"],
                 navigation_calibration["collision_logit_bias"])),
        c_array(fusion["mean"]), c_array(fusion["scale"]), c_array(fusion["weight"]),
        fusion["bias"], fusion["threshold"],
        navigation_calibration["validation"]["threshold_metrics"]["threshold"],
        *corner_thresholds,
    )
    (destination / "inc" / "gap8_perception_output.h").write_text(header)
    weights = sorted(path.name for path in (destination / "hex").glob("*_weights.hex"))
    network_mk = """# Generated ESPNet/DroNet/gate DORY package.
CORE ?= 8
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM
APP_SRCS += $(wildcard $(NETWORK_DIR)/src/*.c)
APP_CFLAGS += -I$(NETWORK_DIR)/inc -DNUM_CORES=$(CORE)
APP_CFLAGS += -DGAP8_MULTITASK_NETWORK=1 -DGAP8_TEMPORAL_NETWORK=1
APP_CFLAGS += -DGAP8_ESPNET_DRONET_GATE=1 -Wno-error -O2 -fno-indirect-inlining -flto
APP_LDFLAGS += -lm -flto
APP_CFLAGS += -DGAP_SDK=1 -DFLASH_TYPE=$(FLASH_TYPE)
APP_CFLAGS += -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS -DFS_READ_FS
"""
    for filename in weights:
        network_mk += "FLASH_FILES += $(NETWORK_DIR)/hex/%s\n" % filename
    network_mk += "READFS_FILES += $(FLASH_FILES)\n"
    (destination / "network.mk").write_text(network_mk)
    manifest = {
        "format": "nanocockpit-espnet-dronet-gate-v1",
        "name": args.name,
        "input": {"dtype": "uint8", "layout": "HWC", "shape": [160, 160, 2],
                  "channel_order": ["previous", "current"]},
        "outputs": {
            "corner_heatmaps": {"shape": [20, 20, 4], "offset": 0, "bytes": 1600},
            "gate_mask": {"shape": [20, 20, 1], "offset": 1600, "bytes": 400},
            "presence": {"shape": [1], "offset": 2000, "bytes": 1},
            "navigation": {"shape": [2], "order": ["yaw", "collision_logit"],
                           "offset": 2001, "bytes": 2},
        },
        "graphs": graphs,
        "deployment_logit_scale": scales,
        "gate_confidence_fusion": fusion,
        "navigation_collision_threshold": training["navigation_collision_threshold"],
        "navigation_output_calibration": navigation_calibration,
        "float_test": {"navigation": training["navigation_test"], "gate": training["gate_test"]},
        "integer_test": json.loads(args.integer_evaluation.read_text()),
        "architecture_selection": json.loads(args.selection_report.read_text()),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
