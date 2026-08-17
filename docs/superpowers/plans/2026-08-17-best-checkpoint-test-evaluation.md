# Best Checkpoint Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locked, reproducible test-split evaluator for the trained P2-CBAM `best.pt` checkpoint and document the technical interpretation in `tailieu.md`.

**Architecture:** Add one standalone CLI script beside the training script. The script performs preflight validation before YOLO construction, registers custom modules before checkpoint load, validates the loaded segmentation architecture, runs exactly one Ultralytics `val(split="test")`, and atomically writes a provenance-rich JSON summary to the actual returned run directory.

**Tech Stack:** Python, pytest, PyYAML, PyTorch, Ultralytics 8.4.13, local CBAM/P2CompatibleSegment26 modules.

---

## Chunk 1: Evaluator And Documentation

### Task 1: Add evaluator tests first

**Files:**
- Create: `D:/PAPER_SPKT/Ham1000_p2_CBAM/tests/test_evaluate_best_test.py`
- Create later: `D:/PAPER_SPKT/Ham1000_p2_CBAM/04_evaluate_best_test.py`

- [ ] **Step 1: Write failing tests for CLI validation and output safety**

Cover `--batch`/`--workers` parsing, default paths, output-name containment, existing output rejection before YOLO construction, missing input rejection before YOLO construction, and lock cleanup on validation failure.

- [ ] **Step 2: Run tests and verify RED**

Run: `C:/Users/zinnn/miniconda3/envs/vungcam_2026/python.exe -m pytest tests/test_evaluate_best_test.py -q`

Expected: FAIL because `04_evaluate_best_test.py` does not exist yet.

- [ ] **Step 3: Write failing tests for YOLO call contract and JSON schema**

Mock the YOLO class and metrics object to verify registration happens before load, `val()` is called once with fixed test arguments, returned `metrics.save_dir` must match expected save dir, `test_metrics.json` is written atomically, and unsupported Ultralytics versions fail.

The JSON tests must explicitly cover:
- schema version, UTC timestamp, absolute checkpoint path, absolute dataset YAML path, checkpoint SHA-256, split/imgsz/batch/workers/seed/save_dir, requested device, resolved device, Python version, Ultralytics version, PyTorch version, CUDA availability/version;
- `requested_val_args` exactly matching the dict passed into `model.val`;
- `effective_val_args` from a mocked validator args object when available;
- fallback `effective_val_args` copied from `requested_val_args` plus `conf: 0.001` and `effective_args_source: "pinned_fallback"` when validator args are unavailable;
- all eight Ultralytics 8.4.13 metric keys: `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `metrics/precision(M)`, `metrics/recall(M)`, `metrics/mAP50(M)`, `metrics/mAP50-95(M)`;
- finite Tensor/NumPy/Python scalars converted to Python numbers, missing metric keys stored as `null`, and missing keys listed in `missing_metrics`;
- atomic write by same-directory temp file plus `os.replace`, including a simulated replace/write failure that leaves no partial `test_metrics.json` and does not hide the original exception.

- [ ] **Step 4: Write failing tests for lock behavior**

Verify an existing sibling `.name.evaluate.lock` fails before YOLO construction, the lock file exists while `model.val()` is running, and lock cleanup happens on load, validation, returned-save-dir mismatch, and JSON failures.

- [ ] **Step 5: Run tests and verify RED**

Run: `C:/Users/zinnn/miniconda3/envs/vungcam_2026/python.exe -m pytest tests/test_evaluate_best_test.py -q`

Expected: FAIL on missing evaluator functions/module.

### Task 2: Implement evaluator

**Files:**
- Create: `D:/PAPER_SPKT/Ham1000_p2_CBAM/04_evaluate_best_test.py`

- [ ] **Step 1: Implement CLI and helpers**

Add typed argument parsers, default path resolution, dataset YAML test-only validation, safe output-name validation, exclusive sibling lock held for the whole run, version check for Ultralytics 8.4.13, device selection, checksum, scalar conversion, metric-key extraction, effective-args extraction/fallback, and atomic JSON writer using same-directory temp file plus `os.replace`.

- [ ] **Step 2: Implement evaluation flow**

Register `CBAM`/`P2CompatibleSegment26`, load `YOLO(weights)`, validate task/head/stride, call `val()` exactly once with fixed arguments, reject returned save-dir mismatch, and write `test_metrics.json`.

- [ ] **Step 3: Run focused tests and fix failures**

Run: `C:/Users/zinnn/miniconda3/envs/vungcam_2026/python.exe -m pytest tests/test_evaluate_best_test.py -q`

Expected: PASS.

### Task 3: Append technical analysis to tailieu.md

**Files:**
- Modify: `D:/PAPER_SPKT/Ham1000_p2_CBAM/tailieu.md`

- [ ] **Step 1: Append one dated non-duplicated UTF-8 section**

Add a `## Ghi chú kỹ thuật ngày 2026-08-17...` section containing:
- CBAM channel attention plus spatial attention preserves tensor shape; P2-P5 strides are 4/8/16/32; prototype uses P3-P5 and remains stride 4; the 200-epoch run shows graph/loss works.
- YAML is correct for scale `n`, but hard-coded CBAM channels 64/128 are not safe for s/m/l/x without adjustment.
- Best validation around epoch 183: mask mAP50 0.8664, mask mAP50-95 0.6874, box mAP50-95 0.7118; epoch 200 is about 0.0017 lower on mask mAP50-95; late validation segmentation loss rises mildly, so use `best.pt`.
- Confusion summary: `nv/bcc/df/vasc` stronger, `mel/akiec` weaker, with `mel->nv` and `akiec->bkl/background` as main risks.
- Class polygon counts: nv 24138, mel 9336, bkl 9220, bcc 4316, akiec 2742, vasc 1188, df 966; NV-excluding augmentation does not balance minority classes with each other.
- Confounders: `optimizer=auto`, rotation 180 degrees, online plus offline augmentation, mixup, and copy-paste require ablation; compare against AdamW/lr 0.001 and a less aggressive dermoscopy augmentation profile.
- Baseline PDF test result: three-seed mask mAP50-95 0.5636±0.0234 must not be compared directly against one-seed validation 0.6874 on 31,880 train images.
- A-D matrix: A=YOLO26n baseline without P2/CBAM; B=P2-only; C=CBAM-only; D=P2+CBAM; same split, data, seeds, hyperparameters, checkpoint-selection rule.
- Rule: choose model/hyperparameters by validation only; test is final reporting only and must not be used to adjust the model.

- [ ] **Step 2: Verify section appears once**

Run: `Select-String -Path tailieu.md -Pattern "Ghi chú kỹ thuật ngày 2026-08-17"`

Expected: exactly one match.

- [ ] **Step 3: Verify final-test precondition text**

Run: `Select-String -Path tailieu.md -Pattern "A=YOLO26n baseline","validation only","test is final reporting"`

Expected: ablation matrix and validation-only checkpoint rule are present before any real final-test evaluation is run.

### Task 4: Final verification

**Files:**
- Verify: `D:/PAPER_SPKT/Ham1000_p2_CBAM/04_evaluate_best_test.py`
- Verify: `D:/PAPER_SPKT/Ham1000_p2_CBAM/tests/test_evaluate_best_test.py`
- Verify: `D:/PAPER_SPKT/Ham1000_p2_CBAM/tailieu.md`

- [ ] **Step 1: Run evaluator tests**

Run: `C:/Users/zinnn/miniconda3/envs/vungcam_2026/python.exe -m pytest tests/test_evaluate_best_test.py -q`

Expected: PASS.

- [ ] **Step 2: Compile evaluator and tests**

Run: `C:/Users/zinnn/miniconda3/envs/vungcam_2026/python.exe -m py_compile 04_evaluate_best_test.py tests/test_evaluate_best_test.py`

Expected: exit 0.

- [ ] **Step 3: Check documentation insertion**

Run: `Select-String -Path tailieu.md -Pattern "A=YOLO26n baseline"`

Expected: line exists in the appended section.

Notes:
- A dedicated git worktree cannot be created because `D:/PAPER_SPKT/Ham1000_p2_CBAM` is not a git repository. Execute in the shared workspace and do not push from this step.
- Do not run the full final test evaluation automatically. The script is prepared for the user to launch with an explicit unique `--name` when the final-test policy is accepted.
