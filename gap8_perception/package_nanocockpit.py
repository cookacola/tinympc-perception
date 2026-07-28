#!/usr/bin/env python3
"""Package a generated DORY GAP8 network for NanoCockpit GAP firmware."""

import argparse
import json
import shutil
from pathlib import Path


CHANNEL_NAMES = (
    "corner_tl",
    "corner_tr",
    "corner_br",
    "corner_bl",
    "obstacle_presence",
    "inverse_range",
    "uncertainty",
    "gate",
)


def c_float(value):
    return "%.10gf" % float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dory-app", type=Path, required=True)
    parser.add_argument("--nemo-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--name", default="gap8-multitask-dory")
    parser.add_argument(
        "--controller-qparams-header",
        type=Path,
        help=(
            "Also emit the model-matched output affine constants consumed by "
            "the STM32 controller."
        ),
    )
    parser.add_argument(
        "--validation-metrics",
        type=Path,
        help="Validation report containing the selected danger threshold.",
    )
    args = parser.parse_args()

    report = json.loads(args.nemo_report.read_text())
    quant = report["packed_output_quantization"]
    if report["packed_channels"] != len(CHANNEL_NAMES):
        raise RuntimeError("NanoCockpit package requires the 8-channel model")
    if report["final_integer_output_shape"] != [1, 8, 40, 40]:
        raise RuntimeError("unexpected output shape")
    if report.get("final_integer_output_nonzero", 0) <= 0:
        raise RuntimeError("refusing to package a degenerate integer output")

    source_dirs = [args.dory_app / name for name in ("src", "inc", "hex")]
    for source in source_dirs:
        if not source.is_dir():
            raise RuntimeError("missing generated DORY directory: %s" % source)

    destination = args.destination / args.name
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_dirs:
        target = destination / source.name
        if target.exists():
            shutil.rmtree(str(target))
        shutil.copytree(str(source), str(target))

    standalone_main = destination / "src" / "gap8_main.c"
    if standalone_main.exists():
        standalone_main.unlink()

    weight_files = sorted(
        path.name for path in (destination / "hex").glob("*_weights.hex")
    )
    if not weight_files:
        raise RuntimeError("no generated DORY weights found")

    network_mk = """# Generated NanoCockpit DORY network package.
CORE ?= 7
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM

APP_SRCS += $(wildcard $(NETWORK_DIR)/src/*.c)
APP_CFLAGS += -I$(NETWORK_DIR)/inc
APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP8_MULTITASK_NETWORK=1
APP_CFLAGS += -Wno-error -O2 -fno-indirect-inlining -flto
APP_LDFLAGS += -lm -flto
APP_CFLAGS += -DGAP_SDK=1
APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS -DFS_READ_FS

"""
    for weight_file in weight_files:
        network_mk += (
            "FLASH_FILES += $(NETWORK_DIR)/hex/%s\n" % weight_file
        )
    network_mk += "READFS_FILES += $(FLASH_FILES)\n"
    (destination / "network.mk").write_text(network_mk)

    offsets = quant["per_channel_spatial_offset"]
    output_scales = quant.get("per_channel_output_scale", [1.0] * 8)
    biases = quant["per_channel_learned_bias"]
    if len(offsets) != 8 or len(output_scales) != 8 or len(biases) != 8:
        raise RuntimeError("invalid per-channel output affine metadata")
    if any(abs(float(scale) - 1.0) > 1.0e-7 for scale in output_scales):
        raise RuntimeError(
            "NanoCockpit/controller ABI does not implement output "
            "equalization; all per-channel output scales must be one"
        )
    danger_threshold = 0.5
    if args.validation_metrics:
        validation = json.loads(args.validation_metrics.read_text())
        recommendation = validation["danger"].get(
            "recommended_threshold_for_recall_0.95"
        )
        if recommendation is None:
            raise RuntimeError(
                "validation has no threshold reaching 0.95 danger recall"
            )
        if not validation["danger"].get(
            "threshold_recommendation_authoritative", False
        ):
            raise RuntimeError("danger threshold must come from validation split")
        danger_threshold = float(recommendation["threshold"])
    channel_enum = "\n".join(
        "  GAP8_OUTPUT_%s = %d," % (name.upper(), index)
        for index, name in enumerate(CHANNEL_NAMES)
    )
    abi_header = """#ifndef GAP8_PERCEPTION_OUTPUT_H
#define GAP8_PERCEPTION_OUTPUT_H

#include <stdint.h>

#define GAP8_INPUT_WIDTH 160
#define GAP8_INPUT_HEIGHT 160
#define GAP8_OUTPUT_WIDTH 40
#define GAP8_OUTPUT_HEIGHT 40
#define GAP8_OUTPUT_CHANNELS 8
#define GAP8_OUTPUT_BYTES (GAP8_OUTPUT_WIDTH * GAP8_OUTPUT_HEIGHT * GAP8_OUTPUT_CHANNELS)
#define GAP8_CONTROL_WIDTH 20
#define GAP8_CONTROL_HEIGHT 20

typedef enum {
%s
} gap8_output_channel_t;

extern const float gap8_output_epsilon;
extern const float gap8_output_spatial_offset[GAP8_OUTPUT_CHANNELS];
extern const float gap8_output_learned_bias[GAP8_OUTPUT_CHANNELS];

static inline uint32_t gap8_output_index(uint32_t x, uint32_t y,
                                         gap8_output_channel_t channel) {
  return (y * GAP8_OUTPUT_WIDTH + x) * GAP8_OUTPUT_CHANNELS + channel;
}

static inline float gap8_output_logit(uint8_t value,
                                      gap8_output_channel_t channel) {
  return value * gap8_output_epsilon
       - gap8_output_spatial_offset[channel]
       + gap8_output_learned_bias[channel];
}

void gap8_decode_corner_argmax(const uint8_t *packed_hwc,
                               float corners_xy[8],
                               uint8_t confidence[4]);
void gap8_pool_control_maps(const uint8_t *packed_hwc,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]);

#endif
""" % channel_enum
    (destination / "inc" / "gap8_perception_output.h").write_text(abi_header)

    abi_source = """#include "gap8_perception_output.h"

const float gap8_output_epsilon = %s;
const float gap8_output_spatial_offset[GAP8_OUTPUT_CHANNELS] = {%s};
const float gap8_output_learned_bias[GAP8_OUTPUT_CHANNELS] = {%s};

void gap8_decode_corner_argmax(const uint8_t *packed_hwc,
                               float corners_xy[8],
                               uint8_t confidence[4]) {
  for (int channel = 0; channel < 4; ++channel) {
    uint8_t best = 0;
    int best_x = 0;
    int best_y = 0;
    for (int y = 0; y < GAP8_OUTPUT_HEIGHT; ++y) {
      for (int x = 0; x < GAP8_OUTPUT_WIDTH; ++x) {
        uint8_t value = packed_hwc[gap8_output_index(
            x, y, (gap8_output_channel_t)channel)];
        if (value > best) {
          best = value;
          best_x = x;
          best_y = y;
        }
      }
    }
    corners_xy[2 * channel] = (best_x + 0.5f) * 4.0f;
    corners_xy[2 * channel + 1] = (best_y + 0.5f) * 4.0f;
    confidence[channel] = best;
  }
}

void gap8_pool_control_maps(const uint8_t *packed_hwc,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]) {
  const int channels[4] = {
    GAP8_OUTPUT_OBSTACLE_PRESENCE, GAP8_OUTPUT_INVERSE_RANGE,
    GAP8_OUTPUT_UNCERTAINTY, GAP8_OUTPUT_GATE
  };
  uint8_t *outputs[4] = {
    obstacle_presence, inverse_range, uncertainty, gate_opening
  };
  for (int oy = 0; oy < GAP8_CONTROL_HEIGHT; ++oy) {
    for (int ox = 0; ox < GAP8_CONTROL_WIDTH; ++ox) {
      for (int map = 0; map < 4; ++map) {
        /* A single dangerous/near/uncertain source pixel must survive
         * downsampling. Gate opening is a permission signal, so require all
         * four source pixels to agree by conservatively min-pooling it. */
        uint8_t pooled = map == 3 ? 255 : 0;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            uint8_t value = packed_hwc[gap8_output_index(
                2 * ox + dx, 2 * oy + dy,
                (gap8_output_channel_t)channels[map])];
            if (map == 3) {
              if (value < pooled) pooled = value;
            } else if (value > pooled) {
              pooled = value;
            }
          }
        }
        outputs[map][oy * GAP8_CONTROL_WIDTH + ox] = pooled;
      }
    }
  }
}
""" % (
        c_float(quant["epsilon"]),
        ", ".join(c_float(value) for value in offsets),
        ", ".join(c_float(value) for value in biases),
    )
    (destination / "src" / "gap8_perception_output.c").write_text(abi_source)

    if args.controller_qparams_header:
        args.controller_qparams_header.parent.mkdir(parents=True, exist_ok=True)
        controller_header = """/* Generated by package_nanocockpit.py. */
#ifndef PERCEPTION_MODEL_QPARAMS_H
#define PERCEPTION_MODEL_QPARAMS_H

#define PERCEPTION_MODEL_CHANNELS 8
#define PERCEPTION_MODEL_OBSTACLE_CHANNEL 4
#define PERCEPTION_MODEL_INVERSE_RANGE_CHANNEL 5
#define PERCEPTION_MODEL_UNCERTAINTY_CHANNEL 6
#define PERCEPTION_MODEL_GATE_CHANNEL 7
#define PERCEPTION_MODEL_DANGER_THRESHOLD %s

static const float perceptionModelOutputEpsilon = %s;
static const float perceptionModelSpatialOffset[PERCEPTION_MODEL_CHANNELS] = {%s};
static const float perceptionModelLearnedBias[PERCEPTION_MODEL_CHANNELS] = {%s};

static inline float perceptionModelOutputLogit(unsigned char q,
                                               int channel) {
  return q * perceptionModelOutputEpsilon
       - perceptionModelSpatialOffset[channel]
       + perceptionModelLearnedBias[channel];
}

#endif
""" % (
            c_float(danger_threshold),
            c_float(quant["epsilon"]),
            ", ".join(c_float(value) for value in offsets),
            ", ".join(c_float(value) for value in biases),
        )
        args.controller_qparams_header.write_text(controller_header)

    adapter = """#ifndef NANOCOCKPIT_DORY_NETWORK_ADAPTER_H
#define NANOCOCKPIT_DORY_NETWORK_ADAPTER_H
#include "gap8_network.h"
#define network_initialize gap8_network_initialize
#define network_terminate gap8_network_terminate
#define network_run gap8_network_run
#define network_run_async gap8_network_run_async
#define network_run_wait gap8_network_run_wait
#define network_run_async_cl gap8_network_run_async_cl
#endif
"""
    (destination / "inc" / "network.h").write_text(adapter)

    manifest = {
        "format": "nanocockpit-dory-network-v1",
        "name": args.name,
        "source_dory_app": str(args.dory_app.resolve()),
        "source_nemo_report": str(args.nemo_report.resolve()),
        "input": {"dtype": "uint8", "layout": "HWC", "shape": [160, 160, 1]},
        "output": {
            "dtype": "uint8",
            "layout": "HWC",
            "shape": [40, 40, 8],
            "channels": list(CHANNEL_NAMES),
            "channel_semantics": {
                "obstacle_presence": (
                    "ABI-compatible name for collision-within-horizon "
                    "probability at the nominal 1.0 m/s training state"
                ),
                "inverse_range": "1 - clipped conservative range / 6 m",
                "uncertainty": "clearance-boundary proximity",
            },
            "epsilon": quant["epsilon"],
            "spatial_offset": offsets,
            "learned_bias": biases,
        },
        "firmware": {
            "cluster_workers": 7,
            "async_entry": "gap8_network_run_async_cl",
            "raw_tensor_uart": False,
            "controller_products": [
                "ordered corner coordinates/confidence",
                "20x20 nominal-speed collision probability",
                "20x20 inverse range",
                "20x20 uncertainty",
                "20x20 gate-opening permission",
            ],
        },
        "controller_qparams_header": (
            str(args.controller_qparams_header.resolve())
            if args.controller_qparams_header
            else None
        ),
        "validation_metrics": (
            str(args.validation_metrics.resolve())
            if args.validation_metrics else None
        ),
        "controller_danger_threshold": danger_threshold,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
