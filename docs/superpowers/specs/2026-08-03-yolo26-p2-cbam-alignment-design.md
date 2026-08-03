# YOLO26 P2-CBAM Alignment Design

## Objective

Convert the project from a nominal YOLOv12-derived configuration into a genuine Ultralytics YOLO26 instance-segmentation model, add a stride-4 P2 segmentation branch, retain CBAM-based high-resolution feature refinement, and make all documentation scientifically consistent with the implementation.

## Scientific Direction

The paper will present an artifact-aware, high-resolution skin-lesion segmentation framework. Its defensible contribution is the combination and experimental evaluation of:

1. artifact-aware multi-view dermoscopic preprocessing;
2. a YOLO26 instance-segmentation network extended with a P2/4 output;
3. CBAM refinement of early, high-resolution features; and
4. class-aware augmentation for the imbalanced diagnostic categories.

The project must not claim that P2, CBAM, DullRazor, or their individual algorithms are novel. It must not claim superiority, metric values, real-time speed, or “first” status without completed comparative experiments.

## Preprocessing Terminology

Use **artifact-aware multi-view dermoscopic preprocessing** as the pipeline name. The pipeline produces an original letterboxed training view and a processed training view consisting of:

- ROI extraction / dermatoscope-vignette removal;
- morphological DullRazor-style hair detection and Telea inpainting;
- Gray-World color constancy; and
- 640 × 640 letterboxing with polygon-coordinate transformation.

Class-aware geometric and photometric augmentation is a subsequent dataset-balancing stage, not part of the deterministic preprocessing chain.

The input provenance dataset is `<project-root>/data/dataset_yolo_fixed_labels/dataset_yolo`, where `<project-root>` is resolved from the processing script location and never from the current working directory. It contains 8,008 training, 998 validation, and 1,007 test JPEG image/label pairs. It is recorded as the source used by Stage 1, but deep analysis for the paper targets the two processed datasets: `<project-root>/data/dataset_yolo_640x640_multiview` and `<project-root>/data/dataset_yolo_aug_p2_cbam`. The current alignment task only performs lightweight existence and split-count checks on the provenance dataset.

Stage 1 creates two training views per source image: `_v1` is letterboxed only, while `_v7` applies ROI/vignette removal, DullRazor-style hair removal, Gray-World normalization, and letterboxing. Validation and test receive only `_v1` letterboxed images. Stage 2 copies the multi-view training set and makes three augmentation attempts for each image whose polygons do not contain class 5 (`nv`). It transforms polygon vertices and rejects transformed polygons below 50 px². A failed transform or an augmentation with no valid remaining polygon is recorded and not retried, so three saved variants are not guaranteed. Validation and test are copied byte-for-byte from Stage 1.

Documentation should describe 31,880 as the observed count of the existing processed training artifact, not a deterministic acceptance target, and avoid claiming exact class balance unless a distribution audit establishes it. For leakage checks, a source identifier is the case-folded filename stem after stripping generated terminal suffixes repeatedly in this order: `_aug<integer>`, then `_v1` or `_v7`; expected HAM10000 identifiers must then match `isic_<digits>`. The implementation fails if normalized identifiers overlap across train, validation, and test.

This alignment task updates scripts and documentation but does not execute either destructive data-generation entry point. Existing `data/` outputs are read-only for audits. Future regeneration remains an explicit user action: each script must resolve its exact destination under `<project-root>/data`, display the source and destination, and require a `--overwrite` flag before replacing an existing destination tree. Processing must emit an audit summary with source/output counts, matched pairs, invalid/skipped files, augmentation attempts and saved outputs, per-class polygon counts, and cross-split overlaps.

## Model Architecture

Start from the installed official `yolo26-seg.yaml` definition and preserve its YOLO26-specific properties:

- `end2end: True`;
- `reg_max: 1`;
- compound scale definitions, with the project filename selecting scale `n`;
- `C3k2` backbone/head blocks;
- `C2PSA` at the deepest backbone stage;
- YOLO26-compatible `SPPF`; and
- `Segment26` as the instance-segmentation head.

Add the P2 path using the official YOLO26 P2 topology: upsample the P3 head feature, concatenate it with the stride-4 backbone feature, refine it, then rebuild the P3, P4, and P5 bottom-up paths. `Segment26` consumes P2, P3, P4, and P5 features.

Insert channel-preserving CBAM modules after the stride-4 and stride-8 backbone feature blocks. The custom CBAM implementation remains registered with Ultralytics and must receive the correct scaled channel counts for YOLO26n.

This project intentionally targets only the `n` scale. After YOLO26 compound scaling, the two CBAM inputs are 64 channels at P2 and 128 channels at P3. The YAML therefore passes explicit arguments `[64]` and `[128]`. The dynamic registration replaces the `CBAM` symbol in both `ultralytics.nn.modules` and `ultralytics.nn.tasks`; startup must assert that both symbols resolve to the project class before YAML parsing. Because CBAM preserves its input shape, Ultralytics' parser fallback correctly keeps `c2 = ch[f]`. Supporting other compound scales requires generating scale-specific CBAM arguments and is outside this change.

The expected final feature strides are `[4, 8, 16, 32]`. Layer indices in the implemented YAML must be documented by comments and verified from the constructed model rather than assumed from this prose.

## Dependency and Weight Compatibility

The reference environment is `ultralytics==8.4.60`, which provides `Segment26`, YOLO26 end-to-end parsing, and the installed official YOLO26 configurations used as the architectural source. Startup must fail with a clear compatibility error if `Segment26` is unavailable or the installed version cannot construct the YAML.

Initialization uses partial transfer from `models/yolo26n-seg.pt`. Unmatched P2, CBAM, and changed four-scale head parameters are expected and remain newly initialized. The training startup must report the transferred-item summary; construction validation must compare source and destination state dictionaries and report matched tensor keys and shape-mismatched/missing destination keys. Missing pretrained weights produce an explicit warning and training from initialization, not an unhandled failure.

## Code and Documentation Changes

- Replace all user-facing `YOLOv12` and `SkinSeg-YOLOv12-P2Attn` references with `YOLO26` and `SkinSeg-YOLO26-P2Attn`.
- Rewrite architecture descriptions to match the actual YOLO26 YAML.
- Replace the stale absolute dataset path with a portable project-relative path. The training script validates the required `images/{train,val,test}` and `labels/{train,val,test}` directories, creates a temporary runtime YAML containing the resolved absolute path, passes that temporary file to Ultralytics, and removes it in a `finally` block. It must not rewrite the project YAML.
- Remove or label unsupported performance and novelty claims as evaluation targets.
- Preserve the existing data and generated run artifacts.

## Validation

Validation must include:

1. source compilation/import checks;
2. construction of the custom YAML through Ultralytics 8.4.60;
3. confirmation that the final module is `Segment26`, `nl == 4`, `reg_max == 1`, end-to-end mode is enabled, and model strides equal `[4, 8, 16, 32]`;
4. a mandatory inference-mode dummy forward on a zero tensor of shape `[1, 3, 256, 256]`, on CUDA when available and CPU otherwise, producing valid finite segmentation outputs without an exception;
5. checks that no case-insensitive `YOLOv12` references remain in authoritative source and documentation files (`*.py`, `*.yaml`, `*.yml`, `*.md`) under the project, while excluding `data/`, `runs/`, `__pycache__/`, weight files, caches, and generated artifacts; and
6. before/after checks that no files under `data/` or existing files under `runs/` were modified.

Data-pipeline tests use small temporary fixtures and verify image/label stem matching, label validation, `_v1`/`_v7` filename derivation, polygon transformation and 50 px² rejection, class-5 augmentation exclusion, three-attempt accounting, byte-identical Stage 1 validation/test copying into Stage 2, and normalized-ID leakage detection. A read-only audit focuses on both processed datasets and reports their image/label counts, polygon validity, class distributions, naming/view composition, and cross-split normalized-ID overlap. It performs only lightweight existence/count checks on the provenance dataset. The audit reports whether the augmented training count equals the observed 31,880 without treating a different future count as an algorithm failure.

Construction failures, CBAM registration failures, invalid dataset layouts, temporary-YAML cleanup failures, and forward failures must raise or report messages that name the failed component and actionable path/version information. A CUDA forward failure may be retried once on CPU, but validation passes only if one device completes the mandatory forward successfully.

Training performance is outside this alignment change. Publication metrics require fresh baseline, ablation, validation, and held-out test runs.
