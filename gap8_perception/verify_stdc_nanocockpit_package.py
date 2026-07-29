#!/usr/bin/env python3
"""Compile and exercise the generated NanoCockpit STDC output decoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


HARNESS = r"""
#include "gap8_perception_output.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CORNER_W 40
#define CORNER_C 4
#define DANGER_OFFSET (40 * 30 * 4)

static int count_value(const uint8_t *values, uint8_t expected) {
  int count = 0;
  for (int i = 0; i < 400; ++i) count += values[i] == expected;
  return count;
}

static void peak(uint8_t *packed, int channel, int x, int y) {
  packed[(y * CORNER_W + x) * CORNER_C + channel] = 255;
}

int main(void) {
  uint8_t packed[GAP8_OUTPUT_BYTES] = {0};
  uint8_t obstacle[400], range[400], uncertainty[400], opening[400];
  gap8_pool_control_maps(packed, obstacle, range, uncertainty, opening);
  if (count_value(obstacle, 255) != 80 ||
      count_value(opening, 255) != 0 ||
      count_value(range, 0) != 400 ||
      count_value(uncertainty, 0) != 400) return 10;

  packed[DANGER_OFFSET + 3 * 10 + 4] = 41;
  gap8_pool_control_maps(packed, obstacle, range, uncertainty, opening);
  if (obstacle[(2 + 6) * 20 + 8] != 0) return 11;
  packed[DANGER_OFFSET + 3 * 10 + 4] = 42;
  gap8_pool_control_maps(packed, obstacle, range, uncertainty, opening);
  if (obstacle[(2 + 6) * 20 + 8] != 255) return 12;

  memset(packed, 255, sizeof(packed));
  memset(packed, 0, DANGER_OFFSET);
  peak(packed, 0, 10, 5);
  peak(packed, 1, 30, 5);
  peak(packed, 2, 30, 25);
  peak(packed, 3, 10, 25);
  gap8_pool_control_maps(packed, obstacle, range, uncertainty, opening);
  int opening_count = count_value(opening, 255);
  if (opening_count < 50 || opening_count > 250) return 13;
  if (count_value(obstacle, 255) != 400) return 14;

  memset(packed, 0, DANGER_OFFSET);
  for (int channel = 0; channel < 4; ++channel) peak(packed, channel, 20, 15);
  gap8_pool_control_maps(packed, obstacle, range, uncertainty, opening);
  if (count_value(opening, 255) != 0) return 15;

  printf("{\"passed\":true,\"valid_gate_opening_cells\":%d}\n", opening_count);
  return 0;
}
"""


def verify(package: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="stdc_decoder_") as temporary:
        root = Path(temporary)
        harness = root / "decoder_harness.c"
        executable = root / "decoder_harness"
        harness.write_text(HARNESS)
        compile_result = subprocess.run(
            [
                "gcc",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(package / "inc"),
                str(package / "src/gap8_perception_output.c"),
                str(harness),
                "-o",
                str(executable),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        run_result = subprocess.run(
            [str(executable)], check=True, text=True, capture_output=True
        )
    result = json.loads(run_result.stdout)
    result.update(
        {
            "format": "nanocockpit-stdc-output-decoder-audit-v1",
            "package": str(package.resolve()),
            "compiler": "gcc -std=c11 -Wall -Wextra -Werror",
            "compile_stderr": compile_result.stderr,
            "checks": [
                "unobserved crop rows are maximally dangerous",
                "danger integer threshold boundary is exact",
                "low-confidence geometry emits no opening",
                "valid confident convex geometry emits an inset opening",
                "degenerate phantom geometry emits no opening",
                "gate permission never mutates the obstacle map in the decoder",
            ],
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
