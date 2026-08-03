import importlib.util
import json
import os
import subprocess
import sys
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
    assert preprocess.DST_ROOT == ROOT / "data/dataset_yolo_640x640_multiview"
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
