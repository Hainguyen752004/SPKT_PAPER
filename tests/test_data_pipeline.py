import importlib.util
import json
import os
import subprocess
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def preserve_real_artifact_inventory():
    def snapshot():
        return {(p.relative_to(ROOT).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                for base in (ROOT / "data", ROOT / "runs") if base.exists()
                for p in base.rglob("*") if p.is_file()}
    before = snapshot()
    yield
    assert snapshot() == before


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preprocess():
    return load("preprocess", "data_processing/01_preprocess.py")


@pytest.fixture(scope="module")
def augment():
    return load("augment", "data_processing/02_augment.py")


@pytest.fixture(scope="module")
def audit():
    return load("audit_dataset", "data_processing/audit_dataset.py")


def make_dataset(root, counts=(1, 1, 1)):
    source = root / "data/dataset_yolo_fixed_labels/dataset_yolo"
    for split, count in zip(("train", "val", "test"), counts):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        for i in range(count):
            stem = f"isic_{1000 * ('train', 'val', 'test').index(split) + i:07d}"
            cv2.imwrite(str(source / "images" / split / f"{stem}.jpg"), np.full((8, 12, 3), 80, np.uint8))
            (source / "labels" / split / f"{stem}.txt").write_text("0 0.1 0.1 0.8 0.1 0.5 0.8\n")
    return source


def test_project_paths_derive_from_script(preprocess, augment):
    assert preprocess.PROJECT_ROOT == ROOT
    assert preprocess.SRC_ROOT == ROOT / "data/dataset_yolo_fixed_labels/dataset_yolo"
    assert preprocess.DST_ROOT == ROOT / "data/dataset_yolo_640x640_multiview_geom_v2"
    assert augment.PROJECT_ROOT == ROOT
    assert augment.SRC_ROOT == ROOT / "data/dataset_yolo_640x640_multiview"
    assert augment.DST_ROOT == ROOT / "data/dataset_yolo_aug_p2_cbam"


def test_preprocess_import_is_independent_of_cwd(tmp_path):
    script = ROOT / "data_processing/01_preprocess.py"
    code = f"import importlib.util; s=importlib.util.spec_from_file_location('p', r'{script}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.PROJECT_ROOT)"
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(ROOT) in result.stdout


@pytest.mark.parametrize("line", [
    "7 0 0 1 0 1 1", "1.5 0 0 1 0 1 1", "0 0 0 1 0 1",
    "0 0 0 1 0 1 1 0.5", "0 nan 0 1 0 1 1", "0 -0.1 0 1 0 1 1",
])
def test_polygon_validation_rejects_malformed(audit, line):
    with pytest.raises(ValueError):
        audit.parse_yolo_lines([line], Path("bad.txt"))


def test_source_validation_rejects_stems_counts_and_unreadable(tmp_path, audit):
    source = make_dataset(tmp_path)
    (source / "labels/train/isic_0000000.txt").rename(source / "labels/train/isic_9999999.txt")
    with pytest.raises(ValueError, match="stem"):
        audit.validate_dataset(source, {"train": 1, "val": 1, "test": 1})
    (source / "labels/train/isic_9999999.txt").rename(source / "labels/train/isic_0000000.txt")
    with pytest.raises(ValueError, match="count"):
        audit.validate_dataset(source, {"train": 2, "val": 1, "test": 1})
    (source / "images/train/isic_0000000.jpg").write_bytes(b"not jpeg")
    with pytest.raises(ValueError, match="readable"):
        audit.validate_dataset(source, {"train": 1, "val": 1, "test": 1})


@pytest.mark.parametrize("name, expected", [
    ("ISIC_0001234.JPG", "isic_0001234"),
    ("ISIC_0001234_v7_aug12.jpg", "isic_0001234"),
    ("isic_0001234_v1_v7_aug2", "isic_0001234"),
])
def test_normalized_source_id(audit, name, expected):
    assert audit.normalized_source_id(name) == expected


def test_normalized_source_id_and_cross_split_rejection(audit):
    with pytest.raises(ValueError, match="isic"):
        audit.normalized_source_id("patient_1_v1.jpg")
    with pytest.raises(ValueError, match="overlap"):
        audit.find_cross_split_overlaps({"train": ["ISIC_1_v7_aug0.jpg"], "val": ["isic_1_v1.jpg"], "test": []}, fail=True)


def test_stage1_view_policy_and_label_math(preprocess):
    assert preprocess.output_stems("isic_1", "train") == ["isic_1_v1", "isic_1_v7"]
    assert preprocess.output_stems("isic_1", "val") == ["isic_1_v1"]
    lines = preprocess.transform_labels_letterbox([(0, [0, 0, 1, 0, 1, 1])], 10, 20, 2, 5, 3, 100)
    assert lines == ["0 0.050000 0.030000 0.250000 0.030000 0.250000 0.430000"]


def test_v7_crossing_polygon_is_intersected_instead_of_coordinatewise_clamped(preprocess):
    items = [(2, [0.0, 0.0, 0.5, 0.5, 0.0, 1.0])]

    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        items, 10, 10, 2, 0, 8, 10, 1.0, 0, 0, new_size=10, pair_id="isic_1"
    )

    assert lines == ["2 0.000000 0.200000 0.300000 0.500000 0.000000 0.800000"]
    assert lines != ["2 0.000000 0.000000 0.300000 0.500000 0.000000 1.000000"]
    assert instances == [{
        "instance_id": "isic_1:0",
        "class_id": 2,
        "source_polygon": [[0.0, 0.0], [5.0, 5.0], [0.0, 10.0]],
        "crop_intersection_polygon": [[2.0, 2.0], [5.0, 5.0], [2.0, 8.0]],
        "canvas_polygon": [[0.0, 2.0], [3.0, 5.0], [0.0, 8.0]],
        "source_area": 25.0,
        "crop_intersection_area": 9.0,
        "canvas_area": 9.0,
        "status": "kept",
        "reason": None,
    }]


def test_v7_fully_outside_polygon_is_dropped_with_stable_audit_reason(preprocess):
    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        [(0, [0.0, 0.0, 0.1, 0.0, 0.0, 0.1])],
        10, 10, 5, 5, 5, 5, 2.0, 0, 0, new_size=10, pair_id="isic_2"
    )

    assert lines == []
    assert instances[0]["instance_id"] == "isic_2:0"
    assert instances[0]["status"] == "dropped"
    assert instances[0]["reason"] == "INSTANCE_OUTSIDE_V7_CROP"
    assert instances[0]["crop_intersection_polygon"] == []
    assert instances[0]["crop_intersection_area"] == 0.0
    assert instances[0]["canvas_polygon"] == []
    assert instances[0]["canvas_area"] == 0.0


def test_v7_diagonal_polygon_outside_crop_is_not_misclassified_by_bbox(preprocess):
    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        [(0, [0, 0, 1, 0, 0, 1])],
        4, 4, 3, 3, 1, 1, 4.0, 0, 0, new_size=4, pair_id="isic_diagonal"
    )

    assert lines == []
    assert instances[0]["reason"] == "INSTANCE_OUTSIDE_V7_CROP"


def test_v7_boundary_only_contact_is_degenerate_not_outside(preprocess):
    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        [(0, [0, 0, 1, 0, 0, 1])],
        4, 4, 2, 2, 1, 1, 4.0, 0, 0, new_size=4, pair_id="isic_touch"
    )

    assert lines == []
    assert instances[0]["reason"] == "INSTANCE_DEGENERATE_AFTER_CLIP"


def test_v7_collapsed_duplicate_polygon_inside_crop_is_degenerate(preprocess):
    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        [(0, [0.6, 0.6, 0.7, 0.7, 0.6, 0.6])],
        10, 10, 5, 5, 5, 5, 2.0, 0, 0, new_size=10, pair_id="isic_collapsed"
    )

    assert lines == []
    assert instances[0]["source_area"] == 0.0
    assert instances[0]["reason"] == "INSTANCE_DEGENERATE_AFTER_CLIP"


def test_v7_label_mapping_uses_effective_integer_raster_scale(preprocess):
    lines, _ = preprocess.transform_labels_crop_and_letterbox(
        [(0, [0.0, 0.0, 1.0, 0.0, 1.0, 1.0])],
        7, 3, 0, 0, 7, 3, 10 / 7, 0, 3, new_size=10, pair_id="isic_3"
    )

    assert lines == ["0 0.000000 0.300000 1.000000 0.300000 1.000000 0.700000"]


@pytest.mark.parametrize("coords,geometry", [
    ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0], (10, 10, 0, 0, 10, 10)),
    ([0, 0, 1, 0, 1, 1, 0.75, 1, 0.75, 0.25, 0.25, 0.25, 0.25, 1, 0, 1],
     (4, 4, 0, 2, 4, 1)),
])
def test_v7_degenerate_or_disconnected_clip_is_dropped(preprocess, coords, geometry):
    orig_w, orig_h, rx, ry, rw, rh = geometry
    scale = min(10 / rw, 10 / rh)
    resized_height = int(rh * scale)
    pad_y = (10 - resized_height) // 2

    lines, instances = preprocess.transform_labels_crop_and_letterbox(
        [(0, coords)], orig_w, orig_h, rx, ry, rw, rh, scale, 0, pad_y,
        new_size=10, pair_id="isic_bad"
    )

    assert lines == []
    assert instances[0]["status"] == "dropped"
    assert instances[0]["reason"] == "INSTANCE_DEGENERATE_AFTER_CLIP"


def test_artifact_proxy_formulas_and_exposed_intermediates(preprocess):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    corrected = preprocess.gray_world_color_constancy(image)
    corrected_with_gains, gains = preprocess.gray_world_color_constancy(image, return_gains=True)
    assert np.array_equal(corrected, corrected_with_gains)
    assert gains == pytest.approx([2.0, 1.0, 2.0 / 3.0])
    legacy_b, legacy_g, legacy_r = cv2.split(image.astype(np.float32))
    legacy_means = [np.mean(channel) for channel in (legacy_b, legacy_g, legacy_r)]
    legacy_gray = sum(legacy_means) / 3.0
    legacy = cv2.merge([
        np.clip(channel * (legacy_gray / mean), 0, 255)
        for channel, mean in zip((legacy_b, legacy_g, legacy_r), legacy_means)
    ]).astype(np.uint8)
    assert np.array_equal(corrected, legacy)

    inpainted = preprocess.dullrazor(image)
    inpainted_with_mask, mask = preprocess.dullrazor(image, return_mask=True)
    assert np.array_equal(inpainted, inpainted_with_mask)
    proxies = preprocess.artifact_proxies(mask, 3, 2, 6, 4, gains)
    assert proxies["hair_mask_coverage"] == np.count_nonzero(mask > 0) / 6
    assert proxies["vignette_crop_ratio"] == 1.0 - 6 / 24
    assert proxies["gray_world_gains"] == pytest.approx(gains)
    assert proxies["gray_world_correction_magnitude"] == pytest.approx(
        math.sqrt((2.0 - 1.0) ** 2 + (1.0 - 1.0) ** 2 + (2.0 / 3.0 - 1.0) ** 2)
    )


def test_build_dataset_writes_paired_metadata_and_default_view_policy(tmp_path, preprocess, monkeypatch):
    source = make_dataset(tmp_path)
    destination = tmp_path / "build"
    monkeypatch.setattr(preprocess, "SRC_ROOT", source)

    records = preprocess.build_dataset(destination)

    persisted = [json.loads(line) for line in (destination / "metadata/transforms.jsonl").read_text().splitlines()]
    assert persisted == records
    for line in (destination / "metadata/transforms.jsonl").read_text().splitlines():
        ordered = json.loads(line, object_pairs_hook=list)
        assert [key for key, _ in ordered] == sorted(key for key, _ in ordered)
    assert [(record["split"], record["view"]) for record in records] == [
        ("train", "v1"), ("train", "v7"), ("val", "v1"), ("test", "v1")
    ]
    train_v1, train_v7 = records[:2]
    assert train_v1["pair_id"] == train_v7["pair_id"] == "isic_0000000"
    assert train_v1["transform"]["crop_box"] == [0.0, 0.0, 12.0, 8.0]
    assert train_v7["transform"]["crop_box"] == [0.0, 0.0, 12.0, 8.0]
    assert train_v1["source"] == train_v7["source"] == "images/train/isic_0000000.jpg"
    assert train_v1["input_instance_count"] == train_v7["input_instance_count"] == 1
    assert train_v1["output_instance_count"] == train_v7["output_instance_count"] == 1
    assert set(train_v7["artifact_proxies"]) == {
        "hair_mask_coverage", "vignette_crop_ratio", "gray_world_gains",
        "gray_world_correction_magnitude",
    }
    assert not (destination / "images/val_v7_eval").exists()
    assert not (destination / "images/test_v7_eval").exists()
    assert (destination / "images/val/isic_0001000_v1.jpg").exists()
    assert (destination / "images/test/isic_0002000_v1.jpg").exists()


def test_generate_v7_eval_uses_separate_non_training_directories(tmp_path, preprocess, monkeypatch):
    source = make_dataset(tmp_path)
    destination = tmp_path / "build"
    monkeypatch.setattr(preprocess, "SRC_ROOT", source)

    records = preprocess.build_dataset(destination, generate_v7_eval=True)

    assert [(record["split"], record["view"]) for record in records] == [
        ("train", "v1"), ("train", "v7"),
        ("val", "v1"), ("val", "v7"),
        ("test", "v1"), ("test", "v7"),
    ]
    assert (destination / "images/val_v7_eval/isic_0001000_v7.jpg").exists()
    assert (destination / "labels/val_v7_eval/isic_0001000_v7.txt").exists()
    assert (destination / "images/test_v7_eval/isic_0002000_v7.jpg").exists()
    assert not (destination / "images/val/isic_0001000_v7.jpg").exists()
    assert not (destination / "images/test/isic_0002000_v7.jpg").exists()


def test_destination_allowlist_rejects_roots_descendants_traversal_and_symlink_escape(tmp_path, preprocess):
    project = tmp_path / "project"
    data = project / "data"
    approved = data / "dataset_yolo_640x640_multiview_geom_v2"
    data.mkdir(parents=True)

    assert preprocess.resolve_destination(approved, project, approved) == approved.resolve()
    unsafe_paths = [
        data,
        project / "outside",
        approved / "child",
        approved / ".." / "wrong",
    ]
    for unsafe in unsafe_paths:
        with pytest.raises(ValueError, match="destination"):
            preprocess.resolve_destination(unsafe, project, approved)

    outside = tmp_path / "outside-target"
    outside.mkdir()
    try:
        approved.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="destination"):
        preprocess.resolve_destination(approved, project, approved)


def test_cli_destination_and_eval_flags(preprocess, tmp_path):
    args = preprocess.parse_args(["--destination", str(tmp_path), "--generate-v7-eval", "--overwrite"])
    assert args.destination == str(tmp_path)
    assert args.generate_v7_eval is True
    assert args.overwrite is True


def test_preprocess_transaction_includes_metadata_replacement_and_rollback(tmp_path, preprocess):
    destination = tmp_path / "data/output"
    metadata = destination / "metadata/transforms.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("old\n")

    def failing_builder(temp):
        path = temp / "metadata/transforms.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("partial\n")
        raise RuntimeError("metadata failure")

    with pytest.raises(RuntimeError, match="metadata failure"):
        preprocess.transactional_build(destination, tmp_path, destination, True, failing_builder)
    assert metadata.read_text() == "old\n"

    def successful_builder(temp):
        path = temp / "metadata/transforms.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("new\n")

    preprocess.transactional_build(destination, tmp_path, destination, True, successful_builder)
    assert metadata.read_text() == "new\n"


@pytest.mark.parametrize("destination_exists", [False, True])
def test_preprocess_transaction_preserves_unowned_legacy_backup(
        tmp_path, preprocess, destination_exists):
    destination = tmp_path / "data/output"
    legacy_backup = destination.with_name(".output.backup")
    legacy_backup.mkdir(parents=True)
    sentinel = legacy_backup / "sentinel.txt"
    sentinel.write_text("unowned")
    if destination_exists:
        destination.mkdir()
        (destination / "old.txt").write_text("old")

    preprocess.transactional_build(
        destination, tmp_path, destination, destination_exists,
        lambda temp: (temp / "new.txt").write_text("new"),
    )

    assert (destination / "new.txt").read_text() == "new"
    assert sentinel.read_text() == "unowned"
    assert not list(destination.parent.glob(".output.backup-*"))


def test_preprocess_transaction_rolls_back_owned_backup_without_touching_legacy(
        tmp_path, preprocess, monkeypatch):
    destination = tmp_path / "data/output"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old")
    legacy_backup = destination.with_name(".output.backup")
    legacy_backup.mkdir()
    sentinel = legacy_backup / "sentinel.txt"
    sentinel.write_text("unowned")
    original_rename = Path.rename

    def fail_build_swap(path, target):
        if ".output.build-" in path.name:
            raise OSError("forced unique-backup rollback")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_build_swap)
    with pytest.raises(OSError, match="forced unique-backup rollback"):
        preprocess.transactional_build(
            destination, tmp_path, destination, True,
            lambda temp: (temp / "new.txt").write_text("new"),
        )

    assert (destination / "old.txt").read_text() == "old"
    assert sentinel.read_text() == "unowned"
    assert not list(destination.parent.glob(".output.backup-*"))


def test_augmentation_policy_attempts_and_area_filter(tmp_path, augment):
    assert augment.should_augment([(5, [(0, 0), (1, 0), (1, 1)])]) is False
    assert augment.should_augment([(2, [(0, 0), (1, 0), (1, 1)])]) is True
    assert augment.poly_area([(0, 0), (10, 0), (10, 4)]) == 20
    attempts = []
    summary = augment.run_augmentation_attempts(lambda k: attempts.append(k) or False)
    assert attempts == [0, 1, 2]
    assert summary == {"augmentation_attempts": 3, "augmentation_saved": 0, "skipped_files": 3}


def test_safe_overwrite_guards(preprocess, tmp_path):
    project = tmp_path.resolve()
    intended = project / "data/stage1"
    intended.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        preprocess.prepare_destination(intended, project, intended, overwrite=False)
    for unsafe in (project / "data", project / "elsewhere", project / "data/wrong"):
        with pytest.raises(ValueError):
            preprocess.prepare_destination(unsafe, project, intended, overwrite=True)


def test_augment_safe_overwrite_guards(augment, tmp_path):
    project = tmp_path.resolve()
    intended = project / "data/stage2"
    intended.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        augment.prepare_destination(intended, project, intended, overwrite=False)
    for unsafe in (project / "data", project / "elsewhere", project / "data/wrong"):
        with pytest.raises(ValueError):
            augment.prepare_destination(unsafe, project, intended, overwrite=True)


def test_augment_train_uses_three_attempt_helper(tmp_path, augment, monkeypatch):
    src, dst = tmp_path / "stage1", tmp_path / "stage2"
    (src / "images/train").mkdir(parents=True)
    (src / "labels/train").mkdir(parents=True)
    cv2.imwrite(str(src / "images/train/isic_1_v1.jpg"), np.ones((10, 10, 3), np.uint8))
    (src / "labels/train/isic_1_v1.txt").write_text("0 0 0 1 0 1 1\n")
    monkeypatch.setattr(augment, "SRC_ROOT", src)
    monkeypatch.setattr(augment, "DST_ROOT", dst)
    calls = []
    monkeypatch.setattr(augment, "run_augmentation_attempts", lambda callback: (calls.extend(range(3)) or {"augmentation_attempts": 3, "augmentation_saved": 0, "skipped_files": 3}))
    summary = augment.augment_train()
    assert calls == [0, 1, 2]
    assert summary["augmentation_attempts"] == 3
    assert (dst / "images/train/isic_1_v1.jpg").exists()
    assert not (ROOT / "data/dataset_yolo_aug_p2_cbam/images/train/isic_1_v1.jpg").exists()


def test_manifest_recursive_and_diff_categories(tmp_path, audit):
    project = tmp_path
    for base in (project / "data", project / "runs"):
        (base / "nested").mkdir(parents=True)
    (project / "data/nested/a").write_bytes(b"aa")
    (project / "runs/nested/b").write_bytes(b"bb")
    before = audit.build_manifest(project)
    assert set(before) == {"data/nested/a", "runs/nested/b"}
    (project / "data/nested/a").write_bytes(b"cc")
    (project / "runs/nested/b").write_bytes(b"long")
    (project / "data/new").write_text("new")
    (project / "runs/removed").write_text("x")
    old = dict(before)
    old["runs/removed"] = audit.file_record(project / "runs/removed")
    (project / "runs/removed").unlink()
    diff = audit.compare_manifests(old, audit.build_manifest(project))
    assert list(diff) == ["added", "removed", "size_changed", "digest_changed"]
    assert all(diff.values())


def test_report_path_safety_and_read_only_audit(tmp_path, audit, monkeypatch):
    source = make_dataset(tmp_path)
    for dataset in ("dataset_yolo_640x640_multiview", "dataset_yolo_aug_p2_cbam"):
        for split in ("train", "val", "test"):
            (tmp_path / "data" / dataset / "images" / split).mkdir(parents=True)
            (tmp_path / "data" / dataset / "labels" / split).mkdir(parents=True)
    for unsafe in (tmp_path / "data/report.json", tmp_path / "runs/report.json", tmp_path / "elsewhere/report.json"):
        with pytest.raises(ValueError):
            audit.validate_output_path(unsafe, tmp_path)
    before = {p.relative_to(tmp_path).as_posix(): audit.file_record(p) for p in source.rglob("*") if p.is_file()}
    report = tmp_path / "audit_reports/report.json"
    manifest = tmp_path / "audit_reports/manifest.json"
    assert audit.main(["--project-root", str(tmp_path), "--report", str(report), "--manifest", str(manifest), "--expected-counts", "1,1,1"]) == 0
    payload = json.loads(report.read_text())
    required = {"source_pairs", "output_images", "output_labels", "invalid_files", "skipped_files", "augmentation_attempts", "augmentation_saved", "class_polygons", "cross_split_overlaps"}
    assert required <= set(payload)
    after = {p.relative_to(tmp_path).as_posix(): audit.file_record(p) for p in source.rglob("*") if p.is_file()}
    assert before == after


def test_processed_audit_reports_unreadable_jpeg(tmp_path, audit):
    make_dataset(tmp_path)
    processed = tmp_path / "data/dataset_yolo_640x640_multiview"
    for split in ("train", "val", "test"):
        (processed / "images" / split).mkdir(parents=True)
        (processed / "labels" / split).mkdir(parents=True)
        (tmp_path / "data/dataset_yolo_aug_p2_cbam/images" / split).mkdir(parents=True)
        (tmp_path / "data/dataset_yolo_aug_p2_cbam/labels" / split).mkdir(parents=True)
    (processed / "images/train/isic_0000000_v1.jpg").write_bytes(b"corrupt")
    (processed / "labels/train/isic_0000000_v1.txt").write_text("0 0 0 1 0 1 1\n")
    payload = audit.audit_project(tmp_path, {"train": 1, "val": 1, "test": 1})
    assert payload["invalid_files"] == [
        {"path": "data/dataset_yolo_640x640_multiview/images/train/isic_0000000_v1.jpg", "reason": "unreadable JPEG"}
    ]


def test_preprocess_does_not_write_label_when_image_write_fails(tmp_path, preprocess, monkeypatch):
    monkeypatch.setattr(preprocess.cv2, "imwrite", lambda *args: False)
    label = tmp_path / "x.txt"
    with pytest.raises(OSError, match="image"):
        preprocess.write_output_pair(tmp_path / "x.jpg", np.zeros((2, 2, 3), np.uint8), label, ["0 0 0 1 0 1 1"])
    assert not label.exists()


def test_augment_write_failure_does_not_leave_label(tmp_path, augment, monkeypatch):
    monkeypatch.setattr(augment, "transform", lambda **kwargs: kwargs)
    monkeypatch.setattr(augment.cv2, "imwrite", lambda *args: False)
    items = [(0, [(0, 0), (1, 0), (1, 1)])]
    with pytest.raises(OSError, match="image"):
        augment.augment_attempt(np.zeros((20, 20, 3), np.uint8), items, 20, 20, "isic_1", 0, tmp_path, tmp_path)
    assert not (tmp_path / "isic_1_aug0.txt").exists()


def test_augment_only_swallows_expected_transform_errors(tmp_path, augment, monkeypatch):
    items = [(0, [(0, 0), (1, 0), (1, 1)])]
    monkeypatch.setattr(augment, "transform", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad geometry")))
    with pytest.warns(RuntimeWarning, match="bad geometry"):
        assert augment.augment_attempt(np.zeros((20, 20, 3), np.uint8), items, 20, 20, "isic_1", 0, tmp_path, tmp_path) is False
    monkeypatch.setattr(augment, "transform", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("configuration bug")))
    with pytest.raises(RuntimeError, match="configuration bug"):
        augment.augment_attempt(np.zeros((20, 20, 3), np.uint8), items, 20, 20, "isic_1", 0, tmp_path, tmp_path)


def test_stage2_preflight_happens_before_overwrite(tmp_path, augment):
    source, destination = tmp_path / "bad-source", tmp_path / "data/stage2"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="source"):
        augment.prepare_pipeline_destination(source, destination, tmp_path, destination, overwrite=True)
    assert marker.read_text() == "keep"


def test_preprocess_unreadable_input_is_fatal(preprocess, monkeypatch):
    monkeypatch.setattr(preprocess.cv2, "imread", lambda path: None)
    with pytest.raises(ValueError, match="unreadable input image"):
        preprocess.read_image_or_raise("broken.jpg")


def test_augment_unreadable_input_is_fatal(augment, monkeypatch):
    monkeypatch.setattr(augment.cv2, "imread", lambda path: None)
    with pytest.raises(ValueError, match="unreadable input image"):
        augment.read_image_or_raise("broken.jpg")


def test_stage2_preflight_rejects_empty_splits_before_overwrite(tmp_path, augment):
    source, destination = tmp_path / "source", tmp_path / "data/stage2"
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        augment.prepare_pipeline_destination(source, destination, tmp_path, destination, overwrite=True)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("fixture_name", ["preprocess", "augment"])
def test_transactional_build_preserves_destination_on_midrun_failure(request, fixture_name, tmp_path):
    module = request.getfixturevalue(fixture_name)
    destination = tmp_path / "data/output"
    destination.mkdir(parents=True)
    (destination / "marker").write_text("usable")
    def failing_builder(temp):
        (temp / "partial").write_text("partial")
        raise ValueError("mid-run failure")
    with pytest.raises(ValueError, match="mid-run"):
        module.transactional_build(destination, tmp_path, destination, True, failing_builder)
    assert (destination / "marker").read_text() == "usable"
    assert not list(destination.parent.glob(".output.build-*"))


@pytest.mark.parametrize("fixture_name, reader", [("preprocess", "read_labels"), ("augment", "read_yolo_seg")])
def test_production_label_readers_are_strict(request, fixture_name, reader, tmp_path):
    module = request.getfixturevalue(fixture_name)
    label = tmp_path / "bad.txt"
    label.write_text("7 0 0 1 0 1 1\n")
    with pytest.raises(ValueError, match="class"):
        getattr(module, reader)(label)


def test_audit_requires_both_processed_roots_and_reports_composition(tmp_path, audit):
    make_dataset(tmp_path)
    with pytest.raises(ValueError, match="processed dataset root"):
        audit.audit_project(tmp_path, {"train": 1, "val": 1, "test": 1})
    for dataset in ("dataset_yolo_640x640_multiview", "dataset_yolo_aug_p2_cbam"):
        root = tmp_path / "data" / dataset
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            suffix = "_v1_aug0" if dataset.endswith("aug_p2_cbam") and split == "train" else "_v1"
            stem = f"isic_{1000 * ('train','val','test').index(split):07d}{suffix}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), np.ones((4, 4, 3), np.uint8))
            (root / "labels" / split / f"{stem}.txt").write_text("0 0 0 1 0 1 1\n")
    payload = audit.audit_project(tmp_path, {"train": 1, "val": 1, "test": 1})
    assert set(payload["datasets"]) == {"dataset_yolo_640x640_multiview", "dataset_yolo_aug_p2_cbam"}
    assert payload["datasets"]["dataset_yolo_aug_p2_cbam"]["view_composition"]["train"]["aug"] == 1
    assert payload["observed_augmented_train_count"] == 1
    assert payload["equals_31880"] is False


@pytest.mark.parametrize("fixture_name", ["preprocess", "augment"])
def test_transactional_swap_failure_rolls_back_and_cleans(request, fixture_name, tmp_path, monkeypatch):
    module = request.getfixturevalue(fixture_name)
    destination = tmp_path / "data/output"
    destination.mkdir(parents=True)
    (destination / "marker").write_text("usable")
    original_rename = Path.rename
    def fail_build_swap(path, target):
        if ".output.build-" in path.name:
            raise OSError("forced swap failure")
        return original_rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_build_swap)
    with pytest.raises(OSError, match="forced swap"):
        module.transactional_build(destination, tmp_path, destination, True,
                                   lambda temp: (temp / "complete").write_text("new"))
    assert (destination / "marker").read_text() == "usable"
    assert not list(destination.parent.glob(".output.build-*"))
    assert not list(destination.parent.glob(".output.backup*"))
