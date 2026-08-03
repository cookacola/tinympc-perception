# Canonical DORY deployment contract

The only canonical deployment graph is the sequential student defined by
`docs/Completed_CNN_Design_Document.md` and
`gap8_perception/model_sequential.py`.

It accepts `[1,1,120,160]` and emits one `[1,12,15,20]` tensor. DORY must lower
one ordered Conv/ReLU graph with folded BatchNorm and a linear terminal `1x1`
convolution. Split corner/danger graphs, dense danger maps, gate masks, inverse
range, and speed-indexed outputs are historical and non-canonical.

A release is complete only after one golden set passes FP32, fake-quant,
integer NeMO, ONNX Runtime, DORY, GVSOC, and physical GAP8 comparison for raw
tensors and decoded corners, offsets, and confidences. Old checkpoints cannot
waive or redefine these gates.

The GAP8 export must use DORY's overflow-safe 64-bit BN/ReLU affine
intermediate for this model (`BNRelu_bits: 64`). The layer-12 folded expression
exceeds signed int32 on the validated fixture even though its final uint8
tensor is valid. The export validator records this requirement and the
generated GAP8 application uses DORY's existing 64-bit templates; this is a
deployment-arithmetic correction, not an architecture change.
