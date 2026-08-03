# Paired Geometry Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace coordinatewise v7 polygon clamping with true polygon clipping, persist reproducible v1/v7 transform metadata, and audit all paired labels before AVC training.

**Architecture:** Add a dependency-free Sutherland–Hodgman clipping module and a metadata/audit module, then integrate both into the transactional preprocessing pipeline. All geometry functions remain pure and unit-testable; the CLI writes one JSONL metadata record per generated view and a deterministic JSON audit report. This is phase 1 of the larger AVC project and must pass before paired dataloading or auxiliary losses are implemented.

**Tech Stack:** Python 3.13, NumPy, OpenCV, pytest, JSON/JSONL, existing YOLO polygon labels.

---

## Chunk 1: Geometry primitives and preprocessing integration

### Task 1: True polygon–rectangle intersection

**Files:**
- Create: `data_processing/paired_geometry.py`
- Create: `tests/test_paired_geometry.py`

- [x] **Step 1: Write failing tests for polygon clipping**

Add tests covering: polygon fully inside; fully outside; crossing each of four crop edges; duplicate adjacent vertices; degenerate output; and area preservation for an inside polygon.

```python
def test_clip_polygon_crossing_crop_rectangle():
    polygon = [(-2.0, 2.0), (5.0, 2.0), (5.0, 8.0), (-2.0, 8.0)]
    clipped = clip_polygon_to_rect(polygon, 0.0, 0.0, 4.0, 10.0)
    assert polygon_area(clipped) == pytest.approx(24.0)
    assert all(0.0 <= x <= 4.0 and 0.0 <= y <= 10.0 for x, y in clipped)
```

- [x] **Step 2: Run the new geometry tests and verify import failure**

Run: `python -m pytest tests/test_paired_geometry.py -q`

Expected: FAIL because `data_processing.paired_geometry` does not exist.

- [x] **Step 3: Implement minimal pure geometry functions**

Implement:

```python
EPSILON = 1e-9

def polygon_area(points): ...
def deduplicate_polygon(points, epsilon=EPSILON): ...
def clip_polygon_to_rect(points, x_min, y_min, x_max, y_max): ...
```

Use Sutherland–Hodgman clipping against left, right, top and bottom. Intersection calculations must handle parallel segments without division by zero. Return `[]` when fewer than three unique vertices remain or area is at most `EPSILON`. Preserve floating-point coordinates; do not round inside the primitive.

- [x] **Step 4: Run geometry tests**

Run: `python -m pytest tests/test_paired_geometry.py -q`

Expected: all Task 1 tests PASS.

- [x] **Step 5: Complete Task 1 review (commit deferred to sync repository)**

```powershell
git add data_processing/paired_geometry.py tests/test_paired_geometry.py
git commit -m "feat: add true polygon crop clipping"
```

### Task 2: Transform metadata and round-trip helpers

**Files:**
- Modify: `data_processing/paired_geometry.py`
- Modify: `tests/test_paired_geometry.py`

- [x] **Step 1: Write failing tests for transform metadata**

Test a `ViewTransform` record containing source width/height, crop box, resize scale, padding and canvas size. Verify source→canvas→source round trip below `1e-5` pixels for non-clipped points, JSON serialization stability, and rejection of invalid crop/scale/canvas values.

```python
def test_view_transform_round_trip():
    transform = ViewTransform(600, 450, (10, 20, 500, 400), 1.2, 8, 16, 640)
    canvas = transform.source_to_canvas([(100.0, 120.0)])
    restored = transform.canvas_to_source(canvas)
    assert restored[0] == pytest.approx((100.0, 120.0), abs=1e-5)
```

- [x] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_paired_geometry.py -q`

Expected: FAIL because `ViewTransform` is undefined.

- [x] **Step 3: Implement immutable metadata model**

Add a frozen dataclass `ViewTransform` with `source_to_canvas`, `canvas_to_source`, `to_dict`, and `from_dict`. Define crop box as `(x, y, width, height)` in source pixels. For v1, crop box is the entire source image. Include `view`, `pair_id`, and preprocessing parameters in the emitted record outside the dataclass.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_paired_geometry.py -q`

Expected: all geometry/metadata tests PASS.

- [x] **Step 5: Complete Task 2 review (commit deferred to sync repository)**

```powershell
git add data_processing/paired_geometry.py tests/test_paired_geometry.py
git commit -m "feat: add reversible paired-view transforms"
```

### Task 3: Integrate true clipping and JSONL metadata into preprocessing

**Files:**
- Modify: `data_processing/01_preprocess.py`
- Modify: `tests/test_data_pipeline.py`
- Modify: `tests/test_paired_geometry.py`

- [x] **Step 1: Write failing integration tests**

Add tests proving:

1. a polygon crossing the crop is intersected rather than vertex-clamped;
2. a polygon fully outside the crop is dropped;
3. v1 and v7 metadata records contain the same `pair_id` and correct transform fields;
4. val/test remain v1-only unless an explicit evaluation flag is used;
5. metadata output is transactionally replaced with the dataset.

- [x] **Step 2: Run targeted tests and verify failure**

Run: `python -m pytest tests/test_data_pipeline.py tests/test_paired_geometry.py -q`

Expected: new tests FAIL against the coordinatewise-clamping implementation.

- [x] **Step 3: Replace crop label transformation**

Update `transform_labels_crop_and_letterbox` to:

- denormalize source polygon;
- intersect it with `[rx, rx+rw] × [ry, ry+rh]` using `clip_polygon_to_rect`;
- drop invalid/zero-area intersections;
- translate by crop origin, letterbox, normalize and serialize;
- return transformed lines plus per-instance audit details rather than silently hiding dropped polygons.

- [x] **Step 4: Persist metadata**

Write `metadata/transforms.jsonl` inside the transactional build directory. Each record must include `pair_id`, `split`, `view`, source relative path, source size, crop box, resize scale, padding, canvas size, preprocessing version, number of input/output instances and dropped-instance reasons. Sort source images before processing and write JSON keys deterministically.

- [x] **Step 5: Add deterministic v7 evaluation option**

Add CLI flag `--generate-v7-eval`. Default behavior remains train v1+v7 and val/test v1-only. With the flag, write val/test processed views to `images/{split}_v7_eval` and matching labels/metadata; never add those folders to the training YAML.

- [x] **Step 6: Run integration tests**

Run: `python -m pytest tests/test_data_pipeline.py tests/test_paired_geometry.py -q`

Expected: PASS.

- [x] **Step 7: Complete Task 3 review (commit in sync repository checkpoint)**

```powershell
git add data_processing/01_preprocess.py data_processing/paired_geometry.py tests/test_data_pipeline.py tests/test_paired_geometry.py
git commit -m "feat: persist valid paired-view geometry"
```

## Chunk 2: Dataset-wide audit and publication gate

### Task 4: Implement paired geometry auditor

**Files:**
- Create: `data_processing/audit_paired_geometry.py`
- Create: `tests/test_paired_geometry_audit.py`

- [x] **Step 1: Write failing auditor tests**

Use temporary miniature datasets to test: missing view; duplicate `pair_id/view`; class-count mismatch; invalid polygon; empty downsampled P2 mask; excessive round-trip error; and a completely valid pair. Assert stable reason codes rather than full prose messages.

- [x] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_paired_geometry_audit.py -q`

Expected: FAIL because the auditor module does not exist.

- [x] **Step 3: Implement audit API and CLI**

Implement `audit_paired_geometry(dataset_root, metadata_path, p2_stride=4)` returning a JSON-serializable report with:

- total and valid pair counts;
- missing/duplicate view counts;
- input/output instance and class correspondence;
- invalid/zero-area/out-of-canvas polygons;
- empty P2-mask count;
- max/mean round-trip error;
- per-reason excluded pair IDs;
- `publication_gate_passed` boolean.

The CLI must write through a temporary file followed by atomic rename and exit nonzero when the gate fails.

- [x] **Step 4: Run auditor tests**

Run: `python -m pytest tests/test_paired_geometry_audit.py -q`

Expected: PASS.

- [x] **Step 5: Complete Task 4 review (commit in sync repository checkpoint)**

```powershell
git add data_processing/audit_paired_geometry.py tests/test_paired_geometry_audit.py
git commit -m "feat: add paired geometry publication gate"
```

### Task 5: Run the real-data audit and document results

**Files:**
- Modify: `tailieu.md`
- Create: `audit_reports/paired_geometry.json`
- Modify: `README.md`

- [ ] **Step 1: Back up the existing processed dataset by using a new versioned destination**

Do not overwrite the user’s current 16.016/31.880-image datasets. Configure the corrected preprocessing output as `data/dataset_yolo_640x640_multiview_geom_v2` until counts and checks pass.

- [ ] **Step 2: Generate corrected data and metadata**

Run:

```powershell
python data_processing/01_preprocess.py --destination data/dataset_yolo_640x640_multiview_geom_v2 --generate-v7-eval
```

Expected: transactional completion with source counts 8.008/998/1.007 and no partial destination on failure. If the exact-destination safety contract requires code support, add and test an explicit versioned destination allow-list before running.

- [ ] **Step 3: Run the publication gate**

Run:

```powershell
python data_processing/audit_paired_geometry.py --dataset data/dataset_yolo_640x640_multiview_geom_v2 --output audit_reports/paired_geometry.json
```

Expected: 8.008 candidate train pairs accounted for; every exclusion has a stable reason; process exits zero only when all gate invariants pass.

- [ ] **Step 4: Run the full regression suite**

Run: `python -m pytest -q`

Expected: all existing and new tests PASS.

- [ ] **Step 5: Update documentation and mandatory log**

Record exact commands, counts, exclusions, round-trip error, output paths and test results in `tailieu.md`. Add reproduction commands and the rule that v7_eval is evaluation-only to `README.md`. Do not claim the gate passed unless the generated JSON says so.

- [ ] **Step 6: Commit Task 5**

```powershell
git add README.md tailieu.md audit_reports/paired_geometry.json
git commit -m "docs: record paired geometry audit"
```

### Task 6: Phase-1 verification and GitHub synchronization

**Files:**
- Verify all files modified above

- [ ] **Step 1: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_paired_geometry.py tests/test_paired_geometry_audit.py tests/test_data_pipeline.py -q
python -m pytest -q
git diff --check
```

Expected: all tests PASS and `git diff --check` returns no errors.

- [ ] **Step 2: Review staged scope**

Confirm no image datasets, `runs/`, caches or new training checkpoints are tracked. Only source, tests, documentation and small JSON audit reports may be committed.

- [ ] **Step 3: Push phase 1**

Push the reviewed commits to `Hainguyen752004/SPKT_PAPER` branch `main`, then record remote commit hash in the original `D:\PAPER_SPKT\Ham1000_p2_CBAM\tailieu.md` and synchronize that final log entry.

---

## Deferred phases

After this plan passes its publication gate, create separate implementation plans for:

1. paired dataset/sampler and exposure-controlled batch scheduling;
2. P2/P3 masked pooling and collapse-resistant AVC/VICReg loss;
3. P2 boundary target/head/loss;
4. custom YOLO26 trainer integration, checkpoint stripping and ablation launcher.

These phases must not begin before true geometry clipping and metadata audit are verified on the real 8.008 training pairs.

---

## Binding execution contracts after plan review

This section is authoritative where an earlier task is less specific.

### Metadata contract

Each JSONL view record stores `pair_id`, `view`, `split`, transform fields and an `instances` list. Every instance contains stable `instance_id="{pair_id}:{source_line_index}"`, class ID, source polygon, crop-intersection polygon, canvas polygon, source/intersection/canvas area, status and stable exclusion reason. This permits deterministic class/instance matching and source-coordinate verification.

Artifact proxies are recorded with these exact formulas:

- `hair_mask_coverage = count(hair_mask > 0) / (crop_width * crop_height)` before inpainting;
- `vignette_crop_ratio = 1 - (crop_width * crop_height) / (source_width * source_height)`;
- `gray_world_correction_magnitude = sqrt((g_b-1)^2 + (g_g-1)^2 + (g_r-1)^2)`, where each unclipped channel gain is `mean_gray / mean_channel`; also store the three gains.

### Publication-gate contract

Dataset-level gate passes only when metadata parses completely, every candidate file/view is accounted for, no duplicate `(pair_id, view)` exists, all serialized output polygons are finite/in-canvas/positive-area, metadata and label files agree, and maximum reversible-transform round-trip error is `<=1e-5` source pixels.

A legitimate v7 crop that removes an instance does not fail the whole dataset gate. That pair is excluded from AVC with `INSTANCE_OUTSIDE_V7_CROP` or `INSTANCE_DEGENERATE_AFTER_CLIP` and remains eligible for ordinary supervised training only when its emitted label is valid. The report must give candidate, AVC-valid and AVC-excluded pair counts and reason-coded IDs. AVC-valid requires both views, identical stable instance IDs/classes, and non-empty P2 masks for every matched instance.

### P2-mask contract

Rasterize the normalized canvas polygon at full integer canvas resolution using OpenCV `fillPoly`, with float vertices converted by round-to-nearest and clipped to `[0, canvas_size-1]`. P2 shape is `(ceil(canvas_height/4), ceil(canvas_width/4))`; downsample the binary mask by `adaptive_max_pool2d`, threshold output at `>0`, and declare it empty only when the resulting foreground count is zero. Tiny positive polygons that survive this rule remain valid; otherwise use `EMPTY_P2_MASK`.

### Destination-safety contract

Task 3 must add `--destination`. The only accepted non-default phase-1 destination is the resolved path `PROJECT_ROOT/data/dataset_yolo_640x640_multiview_geom_v2`. Reject the project data root, traversal/outside paths, arbitrary descendants and symlink-resolved escapes. Reject an existing destination unless `--overwrite` is explicit. Tests must cover every rejection before real-data generation. The existing datasets are never overwritten in phase 1.

### Git execution contract

Implementation and commits occur in `D:\PAPER_SPKT\SPKT_PAPER_sync` or a Git worktree created from it. Dataset-wide commands run against `D:\PAPER_SPKT\Ham1000_p2_CBAM` with code synchronized only after tests. Before every commit, copy the reviewed source/test/docs files into the Git tree, run tests there, inspect staged scope, and exclude datasets/runs/caches. Mandatory logs are first written to the original `tailieu.md`, then synchronized and committed.

### Conference scope contract

This plan remains mandatory data-quality infrastructure for the conference version. It does not require boundary supervision or the complete Q1 extension. After geometry passes, conference implementation prioritizes paired-control and AVC; boundary supervision may be deferred unless stable ablations are available before submission.
