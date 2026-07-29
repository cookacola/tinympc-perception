#!/usr/bin/env python3
"""Package two generated STDC DORY graphs as one NanoCockpit network."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


GRAPH_OUTPUT_BYTES = {"corner": 4 * 30 * 40, "danger": 1 * 8 * 10}
COMMON_SOURCES = {
    "dory_dma.c",
    "mem.c",
    "net_utils.c",
    "pulp_nn_add.c",
    "pulp_nn_avgpool.c",
    "pulp_nn_conv_Co_parallel.c",
    "pulp_nn_conv_HoWo_parallel.c",
    "pulp_nn_conv_Ho_parallel.c",
    "pulp_nn_depthwise_3x3_s1.c",
    "pulp_nn_depthwise_generic.c",
    "pulp_nn_depthwise_generic_less_4_weights.c",
    "pulp_nn_linear.c",
    "pulp_nn_linear_out_32.c",
    "pulp_nn_matmul.c",
    "pulp_nn_maxpool.c",
    "pulp_nn_pointwise_Co_parallel.c",
    "pulp_nn_pointwise_HoWo_parallel.c",
    "pulp_nn_pointwise_Ho_parallel.c",
    "pulp_nn_utils.c",
}


def c_float(value):
    return "%.10gf" % float(value)


def normalized_generated_text(text: str) -> str:
    """Keep generated sources deterministic and acceptable to git diff --check."""
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def namespace_text(text: str, graph: str) -> str:
    text = text.replace("gap8_", "stdc_%s_" % graph)
    text = text.replace("GAP8_", "STDC_%s_" % graph.upper())
    # Stock DORY reserves several megabytes per graph despite these graphs
    # needing only tens of kilobytes. Bound both persistent L3 workspaces so
    # the pair fits simultaneously in AI-deck HyperRAM.
    text = re.sub(r"#define L3_WEIGHTS_SIZE \d+", "#define L3_WEIGHTS_SIZE 262144", text)
    text = re.sub(r"#define L3_INPUT_SIZE \d+", "#define L3_INPUT_SIZE 131072", text)
    text = re.sub(r"#define L3_OUTPUT_SIZE \d+", "#define L3_OUTPUT_SIZE 131072", text)
    return normalized_generated_text(text)


def copy_namespaced_graph(source: Path, destination: Path, graph: str):
    for path in sorted((source / "src").glob("gap8_*.c")):
        if path.name == "gap8_main.c":
            continue
        name = path.name.replace("gap8_", "stdc_%s_" % graph)
        (destination / "src" / name).write_text(
            namespace_text(path.read_text(), graph)
        )
    for path in sorted((source / "inc").glob("gap8_*.h")):
        name = path.name.replace("gap8_", "stdc_%s_" % graph)
        (destination / "inc" / name).write_text(
            namespace_text(path.read_text(), graph)
        )
    for path in sorted((source / "hex").glob("gap8_*_weights.hex")):
        name = path.name.replace("gap8_", "stdc_%s_" % graph)
        shutil.copy2(path, destination / "hex" / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corner-app", type=Path)
    parser.add_argument("--danger-app", type=Path)
    parser.add_argument("--encoder-app", type=Path)
    parser.add_argument("--corner-head-app", type=Path)
    parser.add_argument("--danger-head-app", type=Path)
    parser.add_argument("--nemo-report", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--name", default="gap8-stdc-real-dory-pair")
    parser.add_argument("--controller-qparams-header", type=Path)
    args = parser.parse_args()

    nemo = json.loads(args.nemo_report.read_text())
    parity = json.loads(args.parity_report.read_text())
    graphs = {item["graph"]: item for item in nemo["graphs"]}
    shared = all(
        (args.encoder_app, args.corner_head_app, args.danger_head_app)
    )
    if shared:
        expected_graphs = {"encoder", "corner_head", "danger_head"}
        graph_apps = {
            "encoder": args.encoder_app,
            "corner_head": args.corner_head_app,
            "danger_head": args.danger_head_app,
        }
    else:
        if not args.corner_app or not args.danger_app:
            raise RuntimeError(
                "provide either pair apps or all three shared apps"
            )
        expected_graphs = {"corner", "danger"}
        graph_apps = {
            "corner": args.corner_app,
            "danger": args.danger_app,
        }
    if set(graphs) != expected_graphs:
        raise RuntimeError("NeMO report graph set does not match package mode")
    if parity.get("held_out_split") != "test":
        raise RuntimeError("danger threshold must come from held-out test data")
    threshold = parity["recommended_integer_danger_threshold_for_recall_0.99"]
    if threshold["recall"] < 0.99:
        raise RuntimeError("packaged threshold does not meet 0.99 danger recall")

    destination = args.destination / args.name
    if destination.exists():
        shutil.rmtree(destination)
    for directory in ("src", "inc", "hex"):
        (destination / directory).mkdir(parents=True)

    common_app = next(iter(graph_apps.values()))
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
    for graph, app in graph_apps.items():
        copy_namespaced_graph(app, destination, graph)

    wrapper_header = """#ifndef STDC_PAIR_NETWORK_H
#define STDC_PAIR_NETWORK_H
#include <stddef.h>
#include "pmsis.h"
void network_initialize(void);
void network_terminate(void);
void network_run_async_cl(void *l2_buffer, size_t l2_buffer_size,
                          void *l2_final_output, int exec, int initial_dir,
                          pi_device_t *cluster, pi_task_t *network_done);
#endif
"""
    (destination / "inc" / "network.h").write_text(wrapper_header)
    if shared:
        wrapper_source = """#include "network.h"
#include "stdc_encoder_network.h"
#include "stdc_corner_head_network.h"
#include "stdc_danger_head_network.h"
#include <string.h>

#define STDC_SHARED_BYTES 38400
#define STDC_CORNER_BYTES 4800
#define STDC_DANGER_BYTES 80

static PI_L2 uint8_t shared_output[STDC_SHARED_BYTES];
static PI_L2 uint8_t corner_output[STDC_CORNER_BYTES];
static PI_L2 uint8_t danger_output[STDC_DANGER_BYTES];
static PI_FC_L1 pi_task_t encoder_done;
static PI_FC_L1 pi_task_t corner_done;
static PI_FC_L1 pi_task_t danger_done;
static void *workspace;
static size_t workspace_size;
static void *combined_output;
static int run_exec;
static int run_initial_dir;
static pi_device_t *run_cluster;
static pi_task_t *user_done;

static void danger_finished(void *arg) {
  (void)arg;
  memcpy(combined_output, corner_output, STDC_CORNER_BYTES);
  memcpy((uint8_t *)combined_output + STDC_CORNER_BYTES,
         danger_output, STDC_DANGER_BYTES);
  pi_task_push(user_done);
}

static void corner_finished(void *arg) {
  (void)arg;
  memcpy(workspace, shared_output, STDC_SHARED_BYTES);
  stdc_danger_head_network_run_async_cl(
      workspace, workspace_size, danger_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&danger_done, danger_finished, NULL));
}

static void encoder_finished(void *arg) {
  (void)arg;
  memcpy(workspace, shared_output, STDC_SHARED_BYTES);
  stdc_corner_head_network_run_async_cl(
      workspace, workspace_size, corner_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&corner_done, corner_finished, NULL));
}

void network_initialize(void) {
  stdc_encoder_network_initialize();
  stdc_corner_head_network_initialize();
  stdc_danger_head_network_initialize();
}

void network_terminate(void) {
  stdc_encoder_network_terminate();
  stdc_corner_head_network_terminate();
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
"""
    else:
        wrapper_source = """#include "network.h"
#include "stdc_corner_network.h"
#include "stdc_danger_network.h"
#include <string.h>

#define STDC_INPUT_BYTES 19200
#define STDC_CORNER_BYTES 4800
#define STDC_DANGER_BYTES 80

static PI_L2 uint8_t saved_input[STDC_INPUT_BYTES];
static PI_L2 uint8_t corner_output[STDC_CORNER_BYTES];
static PI_L2 uint8_t danger_output[STDC_DANGER_BYTES];
static PI_FC_L1 pi_task_t corner_done;
static PI_FC_L1 pi_task_t danger_done;
static void *workspace;
static size_t workspace_size;
static void *combined_output;
static int run_exec;
static int run_initial_dir;
static pi_device_t *run_cluster;
static pi_task_t *user_done;

static void danger_finished(void *arg) {
  (void)arg;
  memcpy(combined_output, corner_output, STDC_CORNER_BYTES);
  memcpy((uint8_t *)combined_output + STDC_CORNER_BYTES,
         danger_output, STDC_DANGER_BYTES);
  pi_task_push(user_done);
}

static void corner_finished(void *arg) {
  (void)arg;
  memcpy(workspace, saved_input, STDC_INPUT_BYTES);
  stdc_danger_network_run_async_cl(
      workspace, workspace_size, danger_output, run_exec, run_initial_dir,
      run_cluster, pi_task_callback(&danger_done, danger_finished, NULL));
}

void network_initialize(void) {
  stdc_corner_network_initialize();
  stdc_danger_network_initialize();
}

void network_terminate(void) {
  stdc_corner_network_terminate();
  stdc_danger_network_terminate();
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
  memcpy(saved_input, l2_buffer, STDC_INPUT_BYTES);
  stdc_corner_network_run_async_cl(
      workspace, workspace_size, corner_output, exec, initial_dir, cluster,
      pi_task_callback(&corner_done, corner_finished, NULL));
}
"""
    (destination / "src" / "stdc_pair_network.c").write_text(wrapper_source)

    corner_q = graphs["corner_head" if shared else "corner"]
    danger_q = graphs["danger_head" if shared else "danger"]
    output_header = """#ifndef GAP8_PERCEPTION_OUTPUT_H
#define GAP8_PERCEPTION_OUTPUT_H
#include <stdint.h>
#define GAP8_INPUT_WIDTH 160
#define GAP8_INPUT_HEIGHT 120
#define GAP8_OUTPUT_BYTES 4880
#define GAP8_CONTROL_WIDTH 20
#define GAP8_CONTROL_HEIGHT 20
void gap8_decode_corner_argmax(const uint8_t *packed,
                               float corners_xy[8],
                               uint8_t confidence[4]);
void gap8_pool_control_maps(const uint8_t *packed,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]);
#endif
"""
    (destination / "inc" / "gap8_perception_output.h").write_text(output_header)
    corner_thresholds = []
    for channel in range(4):
        logit = -1.0986122886681098  # logit(0.25)
        value = (
            logit
            + corner_q["output_offset"][channel]
            - corner_q["learned_bias"][channel]
        ) / corner_q["output_epsilon"]
        corner_thresholds.append(max(0, min(255, int(value) + 1)))
    danger_logit = __import__("math").log(
        threshold["threshold"] / (1.0 - threshold["threshold"])
    )
    danger_q_threshold = max(
        0,
        min(
            255,
            int(
                (
                    danger_logit
                    + danger_q["output_offset"][0]
                    - danger_q["learned_bias"][0]
                )
                / danger_q["output_epsilon"]
            )
            + 1,
        ),
    )
    output_source = """#include "gap8_perception_output.h"
#include <string.h>

#define CORNER_W 40
#define CORNER_H 30
#define CORNER_C 4
#define DANGER_W 10
#define DANGER_H 8
#define DANGER_OFFSET (CORNER_W * CORNER_H * CORNER_C)
static const uint8_t corner_threshold[4] = {%s};
#define DANGER_Q_THRESHOLD %d

static float cross2(float ax, float ay, float bx, float by,
                    float px, float py) {
  return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

void gap8_decode_corner_argmax(const uint8_t *packed,
                               float corners_xy[8],
                               uint8_t confidence[4]) {
  for (int c = 0; c < 4; ++c) {
    uint8_t best = 0;
    int bx = 0, by = 0;
    for (int y = 0; y < CORNER_H; ++y) {
      for (int x = 0; x < CORNER_W; ++x) {
        uint8_t value = packed[(y * CORNER_W + x) * CORNER_C + c];
        if (value > best) { best = value; bx = x; by = y; }
      }
    }
    corners_xy[2 * c] = (bx + 0.5f) * 4.0f;
    corners_xy[2 * c + 1] = (by + 0.5f) * 4.0f + 20.0f;
    confidence[c] = best;
  }
}

void gap8_pool_control_maps(const uint8_t *packed,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]) {
  const uint8_t *danger = packed + DANGER_OFFSET;
  float corners[8];
  uint8_t confidence[4];
  gap8_decode_corner_argmax(packed, corners, confidence);
  memset(inverse_range, 0, 400);
  memset(uncertainty, 0, 400);
  memset(gate_opening, 0, 400);
  /* Nearest conservative expansion maps the 10x8 output into the central
   * 20x16 control rows. Top/bottom cropped regions are marked maximally
   * dangerous because the CNN did not observe them. */
  memset(obstacle_presence, 255, 400);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 20; ++x) {
      obstacle_presence[(y + 2) * 20 + x] =
          danger[(y / 2) * DANGER_W + x / 2] >= DANGER_Q_THRESHOLD
              ? 255 : 0;
    }
  }

  /* A gate permission map is emitted only after confidence, ordering,
   * convexity, area, and aspect checks. Invalid geometry is a safe no-op. */
  for (int c = 0; c < 4; ++c) {
    if (confidence[c] < corner_threshold[c]) return;
  }
  if (!(corners[0] < corners[2] && corners[6] < corners[4] &&
        corners[1] < corners[7] && corners[3] < corners[5])) return;
  float signed_area2 = 0.0f;
  float sign = 0.0f;
  for (int edge = 0; edge < 4; ++edge) {
    int next = (edge + 1) & 3;
    int after = (edge + 2) & 3;
    signed_area2 += corners[2 * edge] * corners[2 * next + 1]
                  - corners[2 * next] * corners[2 * edge + 1];
    float side = cross2(
        corners[2 * edge], corners[2 * edge + 1],
        corners[2 * next], corners[2 * next + 1],
        corners[2 * after], corners[2 * after + 1]);
    if (edge == 0) sign = side;
    if (side * sign <= 0.0f) return;
  }
  float area = signed_area2 < 0.0f ? -0.5f * signed_area2
                                  : 0.5f * signed_area2;
  if (area < 128.0f || area > 23000.0f) return;
  float width = 0.5f * (
      (corners[2] - corners[0]) + (corners[4] - corners[6]));
  float height = 0.5f * (
      (corners[7] - corners[1]) + (corners[5] - corners[3]));
  if (width <= 0.0f || height <= 0.0f ||
      width / height < 0.35f || width / height > 2.85f) return;
  float cx = 0.0f, cy = 0.0f, inset[8];
  for (int c = 0; c < 4; ++c) {
    cx += 0.25f * corners[2 * c];
    cy += 0.25f * corners[2 * c + 1];
  }
  for (int c = 0; c < 4; ++c) {
    inset[2 * c] = 0.88f * corners[2 * c] + 0.12f * cx;
    inset[2 * c + 1] = 0.88f * corners[2 * c + 1] + 0.12f * cy;
  }
  for (int y = 0; y < 20; ++y) {
    for (int x = 0; x < 20; ++x) {
      float px = (x + 0.5f) * 8.0f;
      float py = (y + 0.5f) * 8.0f;
      int inside = 1;
      for (int edge = 0; edge < 4; ++edge) {
        int next = (edge + 1) & 3;
        float side = cross2(
            inset[2 * edge], inset[2 * edge + 1],
            inset[2 * next], inset[2 * next + 1], px, py);
        if (side * sign < 0.0f) { inside = 0; break; }
      }
      if (inside) gate_opening[y * 20 + x] = 255;
    }
  }
}
""" % (
        ", ".join(str(value) for value in corner_thresholds),
        danger_q_threshold,
    )
    (destination / "src" / "gap8_perception_output.c").write_text(output_source)

    weights = sorted(path.name for path in (destination / "hex").glob("*_weights.hex"))
    network_mk = """# Generated STDC DORY package.
CORE ?= 7
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM
APP_SRCS += $(wildcard $(NETWORK_DIR)/src/*.c)
APP_CFLAGS += -I$(NETWORK_DIR)/inc
APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP8_MULTITASK_NETWORK=1
APP_CFLAGS += -DGAP8_STDC_PAIR_NETWORK=1
%s
APP_CFLAGS += -Wno-error -O2 -fno-indirect-inlining -flto
APP_LDFLAGS += -lm -flto
APP_CFLAGS += -DGAP_SDK=1 -DFLASH_TYPE=$(FLASH_TYPE)
APP_CFLAGS += -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS -DFS_READ_FS
""" % ("APP_CFLAGS += -DGAP8_STDC_SHARED_NETWORK=1" if shared else "")
    for filename in weights:
        network_mk += "FLASH_FILES += $(NETWORK_DIR)/hex/%s\n" % filename
    network_mk += "READFS_FILES += $(FLASH_FILES)\n"
    (destination / "network.mk").write_text(network_mk)

    manifest = {
        "format": (
            "nanocockpit-stdc-shared-dory-v1"
            if shared
            else "nanocockpit-stdc-dory-pair-v1"
        ),
        "name": args.name,
        "input": {"dtype": "uint8", "layout": "HWC", "shape": [120, 160, 1]},
        "outputs": {
            "corner": {"shape": [30, 40, 4], "bytes": 4800},
            "danger": {"shape": [8, 10, 1], "bytes": 80},
        },
        "integer_affine": {
            name: {
                "epsilon": graphs[
                    name + "_head" if shared else name
                ]["output_epsilon"],
                "offset": graphs[
                    name + "_head" if shared else name
                ]["output_offset"],
                "learned_bias": graphs[
                    name + "_head" if shared else name
                ]["learned_bias"],
            }
            for name in ("corner", "danger")
        },
        "danger_probability_threshold": threshold["threshold"],
        "danger_threshold_metrics": threshold,
        "memory": {
            "proven_peak_directional_allocator_bytes": (
                153984 if shared else 154256
            ),
            "strict_allocator_required_bytes": (
                153985 if shared else 154257
            ),
            "nanocockpit_workspace_configured_bytes": (
                180000 if shared else 160000
            ),
            "linked_static_l2_bytes": 225676 if shared else 207404,
            "dory_max_l1_tile_bytes": 36289,
        },
        "source_reports": {
            "nemo": str(args.nemo_report.resolve()),
            "parity": str(args.parity_report.resolve()),
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.controller_qparams_header:
        args.controller_qparams_header.parent.mkdir(parents=True, exist_ok=True)
        args.controller_qparams_header.write_text(
            """/* Generated by package_stdc_pair_nanocockpit.py. */
#ifndef PERCEPTION_MODEL_QPARAMS_H
#define PERCEPTION_MODEL_QPARAMS_H
#define PERCEPTION_MODEL_DANGER_ONLY 1
#define PERCEPTION_MODEL_DANGER_THRESHOLD %sf
#endif
""" % threshold["threshold"]
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
