# NanoCockpit GAP8 deployment

NanoCockpit's GAP firmware is the production runtime. The standalone DORY
application is the cycle-accurate golden-checksum harness.

## Runtime contract

- Input: native HM01B0 160×160 monochrome `uint8`, HWC.
- Integer output: 40×40×8 `uint8`, HWC.
- Channels: TL, TR, BR, BL; nominal collision; inverse range; uncertainty;
  gate opening.
- Collision, inverse range, and uncertainty use conservative max-pooling.
- Gate opening is a permission signal and uses conservative min-pooling.
- The controller transport is wire version 2. Four uint4 values are packed per
  10×10 cell into a 222-byte header/payload/CRC packet.

The image CNN predicts nominal 1 m/s collision probability. TinyMPC combines
its decoded logit with live speed, horizon, latency, map age, inverse range,
and uncertainty. The generated manifest and headers define:

```text
logit[c] = q[c] * epsilon - spatial_offset[c] + learned_bias[c]
```

The selected graph uses NeMO graph traversal seed 9. The export script records
the seed and refuses to package a model unless 32-image integer parity is
within the checked corner and danger limits.

## Gate versus solid-obstacle handling

The physical gate frame remains collision geometry. The controller treats an
opening as traversable only when all of these hold:

1. ordered gate corners pass the gate PnP validity checks;
2. corners and dense maps carry the same echoed capture tick;
3. the cell lies inside the configurable inset gate polygon;
4. min-pooled gate-opening confidence exceeds the threshold;
5. uncertainty is below its threshold;
6. predicted free range extends beyond the known gate plane.

Thus a near object placed in the gate opening does not receive the gate
permission. Runtime parameters `percept.gateOpen`, `gateThr`, `gateInset`,
`gateCap`, `gateRange`, and `gateUnc` control this behavior.

Exactly three confident ordered corners may recover the fourth with the
affine quadrilateral relation. The recovered point must remain inside the
observed crop and the completed polygon must pass the same ordering,
convexity, area, and aspect checks as a four-corner detection. Two missing
corners remain invalid. The decoder returns the recovered point to the
landmark/PnP path and uses a tighter inset for its gate-opening permission map;
the host-compiled audit measures 64 permission cells for the recovered test
gate versus 81 for the fully observed gate.

## Package and build

```bash
cd ~/isaacsim-workspace
python3 gap8_perception/package_nanocockpit.py \
  --dory-app \
    workspace/gap8_rollout_nemo_dory_hm01b0_v4_epoch22_seed9/gap8_application \
  --nemo-report \
    workspace/gap8_rollout_nemo_dory_hm01b0_v4_epoch22_seed9/integer/nemo_export_report.json \
  --destination \
    ~/tinympc-nanocockpit/src/gap/examples/pulp-frontnet/app/networks \
  --controller-qparams-header \
    ~/tinympc-crazyflie/apps/controller_tinympc_eigen/src/perception_model_qparams.h \
  --validation-metrics \
    workspace/gap8_rollout_float_hm01b0_v4/evaluation/validation_metrics.json

source ~/gap_sdk_dory/configs/ai_deck.sh
cd ~/tinympc-nanocockpit/src/gap/examples/pulp-frontnet
make clean NETWORK_NAME=gap8-multitask-dory
make build NETWORK_NAME=gap8-multitask-dory
```

The current NanoCockpit build uses 172,692 bytes of L2 (32.94%) and produces:

```text
BUILD/GAP8_V2/GCC_RISCV/pulp_frontnet
```

Build the matching STM32 firmware with the official builder:

```bash
apptainer exec \
  --bind ~/tinympc-crazyflie:~/tinympc-crazyflie \
  --pwd ~/tinympc-crazyflie/apps/controller_tinympc_eigen \
  ~/containers/bitcraze-builder.sif \
  bash -lc 'make clean && make -j8'
```

The current output is
`~/tinympc-crazyflie/apps/controller_tinympc_eigen/build/cf21bl.bin`.

## Verification boundary

The following are software evidence:

- integer parity gate passes;
- DORY parses and tiles the generated graph;
- NanoCockpit firmware compiles and links;
- Crazyflie firmware compiles and links;
- 27 perception and 14 controller host tests pass.

GVSOC is complete only when its log contains
`Checking final layer: Checksum OK`. Physical flashing, measured end-to-end
latency, tethered flight, and closed-loop A–F ablations remain separate
hardware validation gates.
