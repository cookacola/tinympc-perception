# Sequential CNN completion audit

This audit applies to the replacement `SequentialSTDCNet`, not the historical
STDC/FPN release. It records the evidence currently available for the CNN
design document's training and deployment requirements.

## Passed or evidenced

- Architecture source: `model_sequential.py`.
- Static contract: input `[N,1,120,160]`, one output `[N,12,15,20]`, ordered
  TL/TR/BR/BL heatmaps followed by four offsets and four confidence fields.
- Resource report: 78,568 parameters and 28,152,000 MACs. The checked-in
  `configs/sequential_architecture.json` is tested against every profiled
  convolution layer.
- Dataset split: `configs/split_shards_75k.json` and
  `/home/cchen/isaacsim-workspace/workspace/gap8_sequential_75k_corpus_audit_v2.json`
  account for 75,000 grouped frames across calibrated, randomized, and hard
  negative scenarios.
- Target source and schema audit are implemented for all 75,000 records. The
  prior archives used inner-opening corners and retained legacy three-speed
  fields, so they are stale under the completed design and must be regenerated
  as `sequential_fixed_normal_v1` before any new training or evaluation.
- Real-flight training: the trained FP32 and QAT checkpoints record 4,438
  real records, with `flight_06,flight_07` selected by the training command and
  sampling weight `0.35`. These records contain gate-corner labels only; they
  do not provide obstacle/danger supervision.
- Historical FP32 and QAT checkpoints exist, but they predate the corrected
  centerline label convention and are not canonical evidence for a new run.
- Integer ONNX: `workspace/gap8_sequential_nemo_v2/integer/sequential_int.onnx`.
- Quantization: `integer/quantization_manifest.json` records unsigned INT8,
  scale `0.06313495337963104`, zero point `0`, and logical score offset `6.0`.
- Provenance: `bridge/bridge_report.json` records the source as
  `workspace/gap8_sequential_student_v2_qat/best_qat.pt`. The previous artifact
  was preserved as `workspace/gap8_sequential_nemo_v2_stale_qat_v2_20260731`.
- ONNX↔NeMo integer parity: `integer/onnx_runtime_parity.json` reports 3,600
  compared elements, zero differing elements, and zero maximum LSB error.
- DORY frontend/backend: `dory_frontend_report.json` reports a single output,
  48 parsed nodes, 24 fused GAP8 nodes, 28,152,000 MACs, a 36,289-byte
  maximum estimated L1 tile against 64,000 bytes, loaded activation checksums,
  and generated C successfully.
- Overflow-safe DORY export: the canonical regeneration uses
  `--bnrelu-bits 64`, recorded in `dory_frontend_report_64bit.json`. This is
  required because layer 12's folded BN affine expression exceeds signed
  int32 for the validated fixture; the installed DORY GAP8 templates emit the
  corresponding `int64_t` intermediate without changing the network.
- Corrected application build: `gap8_application_64/BUILDcore8` builds with
  60,716 B L2 usage and the expected 3,600-byte uint8 terminal tensor.
- Layer-bisect diagnostic: `bisect_sequential_checksums.py` parses the ordered
  24-layer checksum stream and identifies the first failing layer; partial
  logs are reported as inconclusive.
- Generated application compilation: the checksum-enabled 8-core image was
  compiled successfully under `gap8_application/BUILDcore8`; the linker
  reported 78,236 B L2 usage (14.92%). Runtime execution was intentionally
  skipped.

## Not claimed

- No checkpoint has yet been trained from the corrected centerline-label,
  legacy-free target archives.

- GVSOC numerical parity remains blocked by the installed simulator aborting
  with a core dump before application output. This is a simulator-launch issue;
  the generated application reaches the GVSOC launcher and does not produce a
  checksum mismatch. The supplied hardware diagnostic evidence reports all 24
  layer CRCs matching ONNX with the same 64-bit affine path.
- Physical GAP8 parity, latency, memory/DMA measurements, energy per inference,
  and closed-loop flight metrics remain unmeasured.
- The available real-flight records do not support real-world danger or
  false-safe claims because they contain no obstacle labels.
