# YOLO26 P2-CBAM Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mislabeled legacy architecture with a genuine YOLO26n instance-segmentation model using P2 and CBAM, make data processing safe and auditable, and align all project documentation with the implementation.

**Architecture:** Derive the network from Ultralytics 8.4.60 `yolo26-seg.yaml`, extend its feature pyramid using the official YOLO26 P2 topology, and feed four scales to `Segment26`. Keep preprocessing and augmentation as separate deterministic/auditable stages rooted at the project directory, and validate both with focused tests plus a read-only dataset audit.

**Tech Stack:** Python 3, PyTorch 2.6, Ultralytics 8.4.60, OpenCV, Albumentations, PyYAML, pytest.

**Execution root:** Run every command from `D:\PAPER_SPKT\Ham1000_p2_CBAM`. In PowerShell, first run `Set-Location -LiteralPath 'D:\PAPER_SPKT\Ham1000_p2_CBAM'`.

---

## File Map

- Modify `models/yolo26n-seg-p2-cbam.yaml`: genuine YOLO26n P2-CBAM segmentation graph.
- Modify `cbam.py`: project CBAM module and invariant validation.
- Modify `03_train_p2_cbam.py`: compatibility checks, safe runtime dataset YAML, partial transfer reporting.
- Modify `data_processing/01_preprocess.py`: authoritative source resolution, validation, safe overwrite, audit output.
- Modify `data_processing/02_augment.py`: safe overwrite, deterministic accounting, audit output.
- Create `data_processing/audit_dataset.py`: non-mutating dataset validation, counts, distributions, leakage report, and preservation manifests.
- Modify `dataset_p2_cbam.yaml`: portable relative dataset root.
- Modify `README.md` and `tailieu.md`: YOLO26 terminology and defensible paper direction.
- Create `tests/test_model_architecture.py`: construction and forward assertions.
- Create `tests/test_data_pipeline.py`: filename, polygon, source validation, leakage, and augmentation-policy tests.

## Chunk 1: Data Pipeline Safety and Auditability

### Task 1: Add deterministic data-pipeline tests

**Files:**
- Create: `tests/test_data_pipeline.py`
- Modify: `data_processing/01_preprocess.py`
- Modify: `data_processing/02_augment.py`

- [ ] Add fixture tests for project-relative source resolution and exact source/destination paths.
- [ ] Add tests that reject mismatched image/label stems, malformed polygons, non-finite/out-of-range coordinates, and unexpected split counts.
- [ ] Add tests that reject unreadable JPEGs, odd coordinate counts, non-integer/out-of-range class IDs, and normalized identifiers outside `isic_<digits>`.
- [ ] Add tests for normalized source IDs: case-fold, remove extension, `_augN`, `_v1`, `_v7`; assert cross-split overlap fails.
- [ ] Add deterministic tests for `_v1`/`_v7` derivation, polygon transformation, the 50 px² filter, class-5 exclusion, and three-attempt accounting. Seed Python/NumPy/Albumentations where applicable.
- [ ] Add safety tests that require refusal when a destination exists without `--overwrite`, reject destinations outside `<project-root>/data`, reject the data root itself, and reject any descendant other than the exact intended destination.
- [ ] Add audit CLI tests that reject report/manifest paths under `data/` or `runs/`, prove the CLI never opens dataset files for writing, enumerate every nested file, and detect added, removed, size-changed, and same-size/content-changed files.
- [ ] Run `python -m pytest tests/test_data_pipeline.py -v`; expect the new tests to fail against the current scripts.
- [ ] Implement focused reusable validation/audit helpers without changing the mathematical transforms.
- [ ] Create `data_processing/audit_dataset.py` with `--project-root`, `--report`, `--manifest`, and `--compare-manifest` arguments. It must open inputs read-only, write reports only under `audit_reports/`, reject report/manifest paths under `data/` or `runs/`, and exit nonzero with changed-path details when comparison finds any added, removed, size-changed, or digest-changed file.
- [ ] Add `--overwrite`; before recursive replacement, resolve the destination and assert it is a strict descendant of `<project-root>/data` and equals the intended output directory.
- [ ] Emit a JSON-compatible summary with keys `source_pairs`, `output_images`, `output_labels`, `invalid_files`, `skipped_files`, `augmentation_attempts`, `augmentation_saved`, `class_polygons`, and `cross_split_overlaps`.
- [ ] Ensure Stage 2 copies Stage 1 validation/test byte-for-byte and never augments them.
- [ ] Re-run `python -m pytest tests/test_data_pipeline.py -v`; expect PASS.

### Task 2: Audit the two processed datasets without regeneration

**Files:**
- Use: `data_processing/audit_dataset.py`
- Check path/counts only: `data/dataset_yolo_fixed_labels/dataset_yolo/**`
- Read only: `data/dataset_yolo_640x640_multiview/**`
- Read only: `data/dataset_yolo_aug_p2_cbam/**`

- [ ] Run `python -m data_processing.audit_dataset --project-root . --report audit_reports/processed.json`; deeply audit only `dataset_yolo_640x640_multiview` and `dataset_yolo_aug_p2_cbam`, reporting image/label equality, polygon validity, naming/view composition, class distributions, and normalized-ID overlap. Check the provenance dataset only for existence and split counts `8008/998/1007`.
- [ ] Confirm the existing augmented train directory contains 31,880 images and report it as an observation.
- [ ] Confirm from the working file inventory that no data-generation command ran and no file under `data/` or `runs/` was created, removed, or rewritten during implementation.

## Chunk 2: Genuine YOLO26n P2-CBAM Segmentation

### Task 3: Write architecture tests first

**Files:**
- Create: `tests/test_model_architecture.py`
- Modify: `models/yolo26n-seg-p2-cbam.yaml`
- Modify: `cbam.py`

- [ ] Add a test that registers the project CBAM in `ultralytics.nn.modules` and `ultralytics.nn.tasks` and asserts symbol identity.
- [ ] Add a construction test asserting YOLO26-specific `C3k2`, `C2PSA`, and final `Segment26` modules exist.
- [ ] Assert the segmentation head has `nl == 4`, `reg_max == 1`, end-to-end enabled, and strides `[4, 8, 16, 32]`.
- [ ] Add a mandatory inference-mode zero-tensor forward at `[1,3,256,256]`; inspect the YOLO26 segmentation result structure and assert all prediction, mask-coefficient, and prototype tensors are finite.
- [ ] Run `python -m pytest tests/test_model_architecture.py -v`; expect failure because the current YAML uses legacy blocks and `Segment`.

### Task 4: Implement the YOLO26n graph

**Files:**
- Modify: `models/yolo26n-seg-p2-cbam.yaml`
- Modify: `cbam.py`

- [ ] Copy the official YOLO26 parameters: `end2end: True`, `reg_max: 1`, and scale definitions.
- [ ] Replace legacy `C2f` stages with the official `C3k2`, `SPPF`, and `C2PSA` topology.
- [ ] Insert channel-preserving custom CBAM after scaled P2 (64 channels) and P3 (128 channels), with commented layer indices.
- [ ] Extend the neck with the official P2 upsample/concat/refinement and bottom-up reconstruction.
- [ ] Connect P2/P3/P4/P5 to `Segment26`.
- [ ] Re-run `python -m pytest tests/test_model_architecture.py -v`; expect PASS on CUDA or a single CPU fallback.

### Task 5: Harden training initialization

**Files:**
- Modify: `03_train_p2_cbam.py`
- Modify: `dataset_p2_cbam.yaml`
- Test: `tests/test_model_architecture.py`

- [ ] Add tests for Ultralytics/`Segment26` compatibility failure and CBAM registration failure.
- [ ] Add tests that portable dataset configuration is not mutated and temporary YAML cleanup occurs after success and exceptions.
- [ ] Add tests for cleanup failure: preserve and re-raise the original training exception while reporting cleanup failure separately; if no original exception exists, raise the cleanup error.
- [ ] Add tests for partial transfer reporting with matched, missing-destination, and shape-mismatched tensors, plus the missing-weight warning path.
- [ ] Run `python -m pytest tests/test_model_architecture.py -v`; expect the newly added initialization tests to fail before implementation.
- [ ] Change `dataset_p2_cbam.yaml` to `path: data/dataset_yolo_aug_p2_cbam`.
- [ ] Resolve and validate all six split directories from the project root.
- [ ] Create the absolute runtime YAML through a collision-safe temporary file and clean it in `finally` without hiding an original exception.
- [ ] Load `models/yolo26n-seg.pt` partially and report matched, missing, and shape-mismatched tensors; warn clearly when weights are unavailable.
- [ ] Re-run `python -m pytest tests/test_model_architecture.py -v`; expect PASS.

## Chunk 3: Scientific Documentation Alignment

### Task 6: Rewrite project terminology and paper direction

**Files:**
- Modify: `README.md`
- Modify: `tailieu.md`
- Modify: docstrings/messages in `cbam.py`, `03_train_p2_cbam.py`, `data_processing/01_preprocess.py`, `data_processing/02_augment.py`

- [ ] Replace `YOLOv12`/`SkinSeg-YOLOv12-P2Attn` with `YOLO26`/`SkinSeg-YOLO26-P2Attn` in authoritative source/docs.
- [ ] Describe the preprocessing as **artifact-aware multi-view dermoscopic preprocessing** and separate it from class-aware augmentation.
- [ ] Document the exact source path and counts, Train `_v1`/`_v7` rules, Val/Test `_v1`-only rule, augmentation policy, polygon transformation, and the observed 31,880-image output.
- [ ] Explain that P2 is supported by official YOLO26 detection topology and this project adapts it to four-scale instance segmentation with CBAM.
- [ ] Remove unverified claims including “first,” “100%,” fixed FPS, mAP, Dice, SOTA superiority, and guaranteed perfect balance.
- [ ] Present the paper direction as artifact robustness plus high-resolution attention-guided lesion segmentation, requiring baseline and ablation experiments.
- [ ] Run `rg -n -i "yolov12|skinseg-yolov12|100%|mAP50 >|Dice.*>|>65 FPS|đầu tiên" -g '*.py' -g '*.yaml' -g '*.yml' -g '*.md' -g '!data/**' -g '!runs/**' -g '!docs/superpowers/**'`; expect no unsupported authoritative references.

### Task 7: Final verification

**Files:**
- Verify all modified source/docs and created tests.
- Preserve `data/**` and `runs/**`.

- [ ] Run `python -m pytest tests -v`; expect all tests PASS.
- [ ] Run `python -m py_compile cbam.py 03_train_p2_cbam.py data_processing/01_preprocess.py data_processing/02_augment.py data_processing/audit_dataset.py`; expect exit code 0.
- [ ] Build the custom model and print the final head class, four strides, parameter count, and GFLOPs as structural information only.
- [ ] Re-run the processed-data audit and confirm no data-generation command ran and no file inventory/count under `data/` or `runs/` changed during implementation.
- [ ] Inspect the diff manually for stale paths, fabricated results, or accidental changes to user artifacts.
- [ ] Report completed changes, verification evidence, remaining experimental work, and the recommended paper title/direction.

The generated JSON reports and manifests remain under `audit_reports/` as verification evidence; they are outside the preserved `data/` and `runs/` trees.

> This directory is not currently a Git repository, so commit steps are intentionally omitted. No source-control initialization is authorized by this plan.
