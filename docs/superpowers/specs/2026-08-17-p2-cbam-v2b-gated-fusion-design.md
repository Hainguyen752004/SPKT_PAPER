# P2-CBAM-v2B GatedFusion design

Date: 2026-08-17

## Goal

Create a new YOLO26n P2-CBAM variant named v2B that adds lightweight gated fusion at the P3 and P2 neck fusion points. The existing benchmarked baseline model must remain unchanged.

## Non-goals

- Do not edit `models/yolo26n-seg-p2-cbam.yaml`.
- Do not retrain automatically.
- Do not add ASPP-lite or Transformer in this variant.
- Do not use the final test metrics to choose hyperparameters. v2B should be selected and tuned using validation results only.

## Rationale

The current P2-CBAM test result is already strong: mask mAP50-95 = 0.7154 and mask mAP50 = 0.9006. The remaining weakness is likely less about coarse lesion localization and more about class confusion and boundary/context after feature fusion, especially `mel -> nv`, `akiec -> bkl/background`, and minority recall.

Therefore v2B targets the neck fusion points where high-resolution skip features and upsampled semantic features meet. A gate after concatenation can learn how much fused information to keep channel-wise and spatially before the following `C3k2` block.

## Architecture change

Add a channel-preserving `GatedFusion` module in `cbam.py`.

Expected behavior:

- input tensor shape `(N, C, H, W)`
- output tensor shape `(N, C, H, W)`
- no channel projection by default
- registerable by `register_cbam()` under both `ultralytics.nn.modules` and `ultralytics.nn.tasks`

Create a new YAML:

`models/yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml`

Placement:

- P3 fusion: after `Concat([-1, 6])`, before its `C3k2`
- P2 fusion: after `Concat([-1, 3])`, before its `C3k2`

The v2B head indices are expected to shift because of the two inserted modules:

- final Segment26 inputs become `[23, 26, 29, 32]`
- prediction scales remain P2/P3/P4/P5
- strides remain `[4, 8, 16, 32]`
- `Segment26` remains `P2CompatibleSegment26`

## Training impact

The old training script can still train the original baseline. v2B training should be run only after validating that the new YAML builds and forwards correctly. If a later turn adds a v2B training entry point, it must preserve the original baseline command and document how pretrained transfer is handled with shifted head indices.

## Tests

Add focused tests before implementation:

1. `GatedFusion` preserves shape and is registered by identity.
2. v1 YAML has no `GatedFusion`; v2B YAML has exactly two `GatedFusion` modules and final head inputs `[23, 26, 29, 32]`.
3. v2B model builds, uses four finite prediction scales, keeps stride `[4, 8, 16, 32]`, and produces stride-4 prototypes on a 256 input.

## Documentation

Append one short implementation note to `tailieu.md` after verification, so the research record keeps the v2B design context for later pipeline diagrams.
