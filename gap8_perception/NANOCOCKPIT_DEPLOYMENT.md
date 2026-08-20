# NanoCockpit deployment contract

NanoCockpit must consume the canonical sequential model, not the historical
dense-danger package.

- Camera input: native 160x160 HM01B0 bytes, center-cropped to rows 20:140.
- GAP8 model input: `1x1x120x160` monochrome.
- GAP8 model output: one `1x12x15x20` integer tensor.
- Channels: TL/TR/BR/BL score fields, four fixed-normal offset fields, then
  four confidence fields.
- STM32 postprocessing: integer corner argmax/refinement; spatial means for
  offset/confidence fields; confidence rejection; safety-margin subtraction;
  body-to-world plane rotation; TinyMPC knot activation and reference shift.
- Low confidence never grants free space.

`output_contract.py` and `include/perception_output_contract.h` are the shared
channel/shape constants. `controller_sequential.py` is the numerical reference
for the required STM32 implementation.

The old NanoCockpit decoder for collision, inverse range, uncertainty, and gate
permission does not match this ABI and must not be packaged with a sequential
checkpoint. Physical integration remains incomplete until the STM32 decoder
matches the reference and passes golden-vector, GVSOC, and hardware parity.
