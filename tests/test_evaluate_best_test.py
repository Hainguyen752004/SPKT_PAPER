import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "04_evaluate_best_test.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_best_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_dataset_yaml(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": "dataset", "test": "images/test", "names": {0: "lesion"}}),
        encoding="utf-8",
    )
    return data_yaml


def write_checkpoint(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    return weights


def clean_run(module, name):
    save_dir = module.resolve_output_dir(name)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    lock_path = module.lock_path_for(save_dir)
    if lock_path.exists():
        lock_path.unlink()
    return save_dir


class FakeHead:
    def __init__(self, stride=(4, 8, 16, 32)):
        self.stride = list(stride)


class FakeInnerModel:
    def __init__(self, head=None, task="segment", stride=(4, 8, 16, 32)):
        self.task = task
        self.model = [head if head is not None else FakeHead(stride)]
        self.stride = list(stride)


class FakeMetrics:
    def __init__(self, save_dir, results_dict=None, validator=None):
        self.save_dir = save_dir
        self.results_dict = results_dict or {}
        if validator is not None:
            self.validator = validator


def test_parse_args_rejects_invalid_batch_and_workers():
    module = load_eval_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--batch", "0"])
    with pytest.raises(SystemExit):
        module.parse_args(["--workers", "-1"])


def test_default_paths_and_safe_output_name(monkeypatch, tmp_path):
    module = load_eval_module()
    fallback = ROOT / "dataset_p2_cbam.yaml"
    args = module.parse_args([])

    assert args.data == fallback
    assert args.name == "SkinSeg_YOLO26_P2_CBAM_Test"
    assert module.resolve_weights_default().name == "best.pt"
    assert module.resolve_output_dir("simple_name") == (ROOT / "runs" / "segment" / "simple_name").resolve()

    for unsafe in ("../escape", "nested/name", r"nested\name", "", "."):
        with pytest.raises(ValueError, match="single directory name"):
            module.resolve_output_dir(unsafe)


def test_missing_inputs_and_existing_output_fail_before_yolo(monkeypatch, tmp_path):
    module = load_eval_module()
    yolo_calls = []
    monkeypatch.setattr(module, "YOLO", lambda weights: yolo_calls.append(weights))
    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")

    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        module.run_evaluation(["--weights", str(tmp_path / "missing.pt"), "--data", str(write_dataset_yaml(tmp_path)), "--name", "NoLoadA"])

    weights = write_checkpoint(tmp_path)
    existing = clean_run(module, "ExistingRun")
    existing.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileExistsError, match="already exists"):
        module.run_evaluation(["--weights", str(weights), "--data", str(write_dataset_yaml(tmp_path)), "--name", "ExistingRun"])

    assert yolo_calls == []


def test_dataset_requires_test_images_and_labels_before_yolo(monkeypatch, tmp_path):
    module = load_eval_module()
    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")
    yolo_calls = []
    monkeypatch.setattr(module, "YOLO", lambda weights: yolo_calls.append(weights))
    dataset = tmp_path / "dataset"
    (dataset / "images" / "test").mkdir(parents=True)
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": "dataset", "test": "images/test"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"labels[/\\]test"):
        module.run_evaluation(["--weights", str(write_checkpoint(tmp_path)), "--data", str(data_yaml), "--name", "MissingLabels"])

    assert yolo_calls == []


def test_yolo_call_contract_json_schema_and_effective_args(monkeypatch, tmp_path):
    module = load_eval_module()
    data_yaml = write_dataset_yaml(tmp_path)
    weights = write_checkpoint(tmp_path)
    run_name = "ContractRun"
    expected_save_dir = clean_run(module, run_name)
    calls = []

    class FakeYOLO:
        def __init__(self, weights_path):
            calls.append(("load", weights_path, module.registration_events.copy()))
            self.model = FakeInnerModel()

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            validator = types.SimpleNamespace(args=types.SimpleNamespace(**kwargs, conf=0.002))
            results = {
                "metrics/precision(B)": 1,
                "metrics/recall(B)": 0.5,
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.25,
                "metrics/precision(M)": 0.9,
            }
            return FakeMetrics(expected_save_dir, results, validator)

    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")
    monkeypatch.setattr(module, "YOLO", FakeYOLO)
    monkeypatch.setattr(module, "P2CompatibleSegment26", FakeHead)
    monkeypatch.setattr(module, "registration_events", [], raising=False)
    monkeypatch.setattr(module, "register_custom_modules", lambda: module.registration_events.append("registered"))

    summary_path = module.run_evaluation([
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--name", run_name,
        "--batch", "3",
        "--workers", "0",
        "--device", "cpu",
        "--seed", "7",
    ])

    assert calls[0] == ("load", str(weights.resolve()), ["registered"])
    val_calls = [call for call in calls if call[0] == "val"]
    assert len(val_calls) == 1
    val_args = val_calls[0][1]
    assert val_args["split"] == "test"
    assert val_args["imgsz"] == 640
    assert val_args["plots"] is True
    assert val_args["save_json"] is True
    assert val_args["iou"] == 0.7
    assert val_args["deterministic"] is True
    assert val_args["exist_ok"] is False
    assert "conf" not in val_args
    assert summary_path == expected_save_dir / "test_metrics.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["checkpoint"] == str(weights.resolve())
    assert summary["data"] == str(data_yaml.resolve())
    assert summary["checkpoint_sha256"]
    assert summary["split"] == "test"
    assert summary["imgsz"] == 640
    assert summary["batch"] == 3
    assert summary["workers"] == 0
    assert summary["seed"] == 7
    assert summary["requested_device"] == "cpu"
    assert summary["resolved_device"] == "cpu"
    assert summary["save_dir"] == str(expected_save_dir)
    assert summary["ultralytics_version"] == "8.4.13"
    assert summary["requested_val_args"] == val_args
    assert summary["effective_val_args"]["conf"] == 0.002
    assert summary["effective_args_source"] == "validator"
    assert summary["metrics"]["metrics/precision(B)"] == 1
    assert summary["metrics"]["metrics/mAP50-95(M)"] is None
    assert "metrics/mAP50-95(M)" in summary["missing_metrics"]


def test_effective_args_fallback_and_version_rejection(monkeypatch, tmp_path):
    module = load_eval_module()
    data_yaml = write_dataset_yaml(tmp_path)
    weights = write_checkpoint(tmp_path)
    run_name = "FallbackRun"
    expected_save_dir = clean_run(module, run_name)

    class FakeYOLO:
        def __init__(self, weights_path):
            self.model = FakeInnerModel()

        def val(self, **kwargs):
            return FakeMetrics(expected_save_dir, {key: 0.1 for key in module.METRIC_KEYS})

    monkeypatch.setattr(module, "YOLO", FakeYOLO)
    monkeypatch.setattr(module, "P2CompatibleSegment26", FakeHead)
    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")
    summary_path = module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", run_name])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["effective_args_source"] == "pinned_fallback"
    assert summary["effective_val_args"]["conf"] == 0.001

    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.14")
    with pytest.raises(RuntimeError, match="ultralytics==8.4.13"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", "BadVersion"])


def test_lock_collision_presence_during_val_and_cleanup_on_failures(monkeypatch, tmp_path):
    module = load_eval_module()
    data_yaml = write_dataset_yaml(tmp_path)
    weights = write_checkpoint(tmp_path)
    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")
    monkeypatch.setattr(module, "P2CompatibleSegment26", FakeHead)

    lock_name = "LockedRun"
    clean_run(module, lock_name)
    lock_path = module.lock_path_for(module.resolve_output_dir(lock_name))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("busy", encoding="utf-8")
    yolo_calls = []
    monkeypatch.setattr(module, "YOLO", lambda weights_path: yolo_calls.append(weights_path))
    with pytest.raises(FileExistsError, match="lock"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", lock_name])
    assert yolo_calls == []
    lock_path.unlink()

    run_name = "CleanupRun"
    expected_save_dir = clean_run(module, run_name)

    class FailingYOLO:
        def __init__(self, weights_path):
            self.model = FakeInnerModel()

        def val(self, **kwargs):
            assert module.lock_path_for(expected_save_dir).exists()
            raise RuntimeError("validation failed")

    monkeypatch.setattr(module, "YOLO", FailingYOLO)
    with pytest.raises(RuntimeError, match="validation failed"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", run_name])
    assert not module.lock_path_for(expected_save_dir).exists()
    assert not (expected_save_dir / "test_metrics.json").exists()


def test_rejects_wrong_head_stride_save_dir_and_atomic_json_failure(monkeypatch, tmp_path):
    module = load_eval_module()
    data_yaml = write_dataset_yaml(tmp_path)
    weights = write_checkpoint(tmp_path)
    monkeypatch.setattr(module.ultralytics, "__version__", "8.4.13")
    monkeypatch.setattr(module, "P2CompatibleSegment26", FakeHead)

    class WrongStrideYOLO:
        def __init__(self, weights_path):
            self.model = FakeInnerModel(stride=(8, 16, 32))

    monkeypatch.setattr(module, "YOLO", WrongStrideYOLO)
    clean_run(module, "WrongStride")
    with pytest.raises(RuntimeError, match="stride"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", "WrongStride"])

    run_name = "WrongSaveDir"
    expected_save_dir = clean_run(module, run_name)

    class WrongSaveYOLO:
        def __init__(self, weights_path):
            self.model = FakeInnerModel()

        def val(self, **kwargs):
            return FakeMetrics(expected_save_dir.parent / "WrongSaveDir2", {key: 0.1 for key in module.METRIC_KEYS})

    monkeypatch.setattr(module, "YOLO", WrongSaveYOLO)
    with pytest.raises(RuntimeError, match="save_dir"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", run_name])
    assert not module.lock_path_for(expected_save_dir).exists()

    run_name = "JsonFailure"
    expected_save_dir = clean_run(module, run_name)

    class JsonYOLO:
        def __init__(self, weights_path):
            self.model = FakeInnerModel()

        def val(self, **kwargs):
            return FakeMetrics(expected_save_dir, {key: 0.1 for key in module.METRIC_KEYS})

    monkeypatch.setattr(module, "YOLO", JsonYOLO)
    monkeypatch.setattr(module.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        module.run_evaluation(["--weights", str(weights), "--data", str(data_yaml), "--name", run_name])
    assert not (expected_save_dir / "test_metrics.json").exists()
    assert not module.lock_path_for(expected_save_dir).exists()
