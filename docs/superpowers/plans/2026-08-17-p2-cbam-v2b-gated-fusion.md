# Plan: P2-CBAM-v2B GatedFusion

> **For anh/em:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan.

## Objective

Implement a separate v2B model variant that adds `GatedFusion` at the P3 and P2 neck fusion points, while keeping the original benchmarked P2-CBAM YAML unchanged.

## Steps

1. Add RED tests
   - Create `tests/test_v2b_gated_fusion.py`.
   - Test `GatedFusion` shape preservation and registration.
   - Test old YAML unchanged and new v2B YAML contract.
   - Test v2B model build/forward/stride/prototype behavior.
   - Run the focused test file and confirm it fails before implementation.

2. Implement `GatedFusion`
   - Add `GatedFusion` to `cbam.py`.
   - Register it in `register_cbam()` under both Ultralytics module namespaces.
   - Keep `CBAM` and `P2CompatibleSegment26` behavior unchanged.

3. Add v2B YAML
   - Create `models/yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml`.
   - Insert `GatedFusion` after the P3 and P2 `Concat` layers.
   - Update downstream concat references and final Segment26 inputs to `[23, 26, 29, 32]`.

4. Verify
   - Run focused v2B tests.
   - Run existing architecture tests.
   - Run Python compile checks for modified Python files.

5. Document
   - Append a concise v2B implementation note to `tailieu.md`.
   - Include exact file names and state that no training/test selection has been done yet.

6. Handoff
   - Report what changed.
   - Provide the next safe step: train v2B on validation protocol only, then compare against baseline.
