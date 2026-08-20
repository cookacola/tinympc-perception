# ONNX models

This directory contains both convenient float inference and the exact split
integer deployment used to generate the GAP8 firmware.

## Float end-to-end model

`espnet_dronet_gate_seed2027_float.onnx` is the easiest model to use with
ONNX Runtime. Its input is a float32 NCHW tensor shaped
`[batch, 2, 160, 160]`, scaled to `[0, 1]`. Channel 0 is the previous frame
and channel 1 is the immediately following frame from the same flight.

Outputs are raw logits:

- `corner_heatmaps`: `[batch, 4, 20, 20]`
- `gate_mask_logits`: `[batch, 1, 20, 20]`
- `gate_presence_logit`: `[batch]`
- `navigation_logits`: `[batch, 2]`, ordered yaw and collision logit

Apply sigmoid to the mask, presence, and collision logits. Navigation output
0 is the yaw regression directly. This ONNX uses opset 13 and supports a
dynamic batch dimension.

## GAP8 integer graph set

The five `*_int.onnx` files are the NeMO integer-domain graphs consumed by
DORY. Run `encoder_int.onnx` first, then feed its `[1, 64, 20, 20]` output to
each head. These ONNX tensors use FLOAT containers for compatibility but their
values represent quantized integers. Input batch size is fixed at one.

Use the quantization constants in the release `manifest.json`, or the matching
NanoCockpit `gap8_perception_output.h`, to decode the integer head outputs.
The split graphs are the authoritative match for the published GAP8 C/hex
firmware and passed exact per-layer GVSOC checks.

## SHA-256

```text
351cdb3357b79eec69b1ae18c70797760b9bbc59338a7b2eafd0f2523519f23f  corner_head_int.onnx
1ba6067ae96b35a01f1d57f402c47486ca250f3ea454f6adc17e95cf10de89d6  encoder_int.onnx
6e9fc06581e29f0681ca63e8630ab21c16c72b9ad9dbfa8646234edee967c7a2  espnet_dronet_gate_seed2027_float.onnx
0aeb470eca12e4312e39cec4a2df8f61a55b793a44f792e52a4d7596c15f1375  gate_head_int.onnx
24fbd15482c83452caa14313d2faa896fb00e304c37d1b496371def8b10b5f4e  navigation_head_int.onnx
886d97f09518f9cdc37df7828bc4096f6d01a1a6e12661889679e5a371399a12  presence_head_int.onnx
```
