import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_p2_cbam", ROOT / "03_train_p2_cbam.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cbam_is_channel_preserving_and_registered_by_identity():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    from cbam import CBAM, register_cbam

    register_cbam()
    assert modules.CBAM is CBAM
    assert tasks.CBAM is CBAM
    layer = CBAM(64, 64)
    output = layer(torch.randn(1, 64, 8, 8))
    assert output.shape == (1, 64, 8, 8)
    with pytest.raises(ValueError, match="preserve channels"):
        CBAM(64, 128)


def test_yaml_is_yolo26n_p2_segmentation_architecture():
    config = yaml.safe_load((ROOT / "models" / "yolo26n-seg-p2-cbam.yaml").read_text(encoding="utf-8"))
    modules = [row[2] for row in config["backbone"] + config["head"]]
    assert config["end2end"] is True
    assert config["reg_max"] == 1
    assert config["scale"] == "n"
    assert config["scales"]["n"] == [0.5, 0.25, 1024]
    assert "C3k2" in modules
    assert "C2PSA" in modules
    assert "Segment26" in modules
    assert modules.count("CBAM") == 2
    assert config["head"][-1][0] == [21, 24, 27, 30]


def test_model_build_and_eval_forward_has_four_finite_scales():
    from cbam import register_cbam
    from ultralytics import YOLO
    from ultralytics.nn.modules import Segment26

    register_cbam()
    model = YOLO(str(ROOT / "models" / "yolo26n-seg-p2-cbam.yaml")).model
    head = model.model[-1]
    assert isinstance(head, Segment26)
    assert head.nl == 4
    assert head.reg_max == 1
    assert head.end2end is True
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 256))
    predictions, auxiliary = output
    detections, prediction_prototypes = predictions
    assert torch.isfinite(detections).all()
    assert torch.isfinite(prediction_prototypes).all()
    assert prediction_prototypes.shape[-2:] == (64, 64)
    for branch in ("one2many", "one2one"):
        assert len(auxiliary[branch]["feats"]) == 4
        assert torch.isfinite(auxiliary[branch]["mask_coefficient"]).all()
        assert torch.isfinite(auxiliary[branch]["proto"]).all()


def test_runtime_dataset_yaml_is_absolute_validated_and_source_unchanged(tmp_path):
    train = load_training_module()
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    source = tmp_path / "dataset.yaml"
    source.write_text(
        yaml.safe_dump({"path": "dataset", "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "x"}}),
        encoding="utf-8",
    )
    original = source.read_bytes()
    runtime = train.create_runtime_dataset_yaml(source)
    try:
        runtime_config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        assert Path(runtime_config["path"]).is_absolute()
        assert source.read_bytes() == original
    finally:
        runtime.unlink()


def test_dataset_validation_reports_missing_label_directory(tmp_path):
    train = load_training_module()
    source = tmp_path / "dataset.yaml"
    source.write_text(yaml.safe_dump({"path": "missing", "train": "images/train", "val": "images/val", "test": "images/test"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"labels[/\\]train"):
        train.create_runtime_dataset_yaml(source)


def test_transfer_report_classifies_synthetic_state_dicts():
    train = load_training_module()
    destination = {"model.0.same": torch.zeros(2), "model.0.different": torch.zeros(3), "model.1.missing": torch.zeros(1)}
    source = {"model.0.same": torch.ones(2), "model.0.different": torch.ones(4), "model.99.extra": torch.ones(1)}
    report, matched = train.build_transfer_report(destination, source)
    assert report["mapped"] == [{"source": "model.0.same", "destination": "model.0.same"}]
    assert report["missing_destination"] == ["model.1.missing"]
    assert report["shape_mismatched"] == [{"source": "model.0.different", "destination": "model.0.different"}]
    assert report["skipped_unmapped"] == ["model.99.extra"]
    assert list(matched) == ["model.0.same"]


def test_semantic_transfer_maps_stock_layers_without_shift_collision():
    train = load_training_module()
    from cbam import register_cbam
    from ultralytics import YOLO

    register_cbam()
    stock = YOLO("yolo26-seg.yaml").model
    custom = YOLO(str(ROOT / "models" / "yolo26n-seg-p2-cbam.yaml")).model
    source = stock.state_dict()
    marker = torch.full_like(source["model.10.cv1.conv.weight"], 0.125)
    source["model.10.cv1.conv.weight"] = marker
    report, matched = train.build_transfer_report(custom.state_dict(), source)
    custom.load_state_dict(matched, strict=False)
    assert torch.equal(custom.state_dict()["model.12.cv1.conv.weight"], marker)
    assert torch.equal(custom.state_dict()["model.10.cv1.conv.weight"], source["model.8.cv1.conv.weight"])
    assert not torch.equal(custom.state_dict()["model.10.cv1.conv.weight"], marker)
    assert {entry["source"] for entry in report["mapped"]} <= {
        key for key in source if int(key.split(".")[1]) <= 16
    }
    assert not any(entry["destination"].startswith("model.19.") for entry in report["mapped"])


def test_missing_pretrained_weights_warns_without_loading(tmp_path):
    train = load_training_module()
    with pytest.warns(UserWarning, match="initializing from YAML"):
        report = train.load_partial_pretrained(object(), tmp_path / "yolo26n-seg.pt")
    assert report == {"mapped": [], "missing_destination": [], "shape_mismatched": [], "skipped_unmapped": []}


def test_packaged_pretrained_path_exists():
    train = load_training_module()
    path = train.pretrained_weights_path()
    assert path == ROOT / "models" / "yolo26n-seg.pt"
    assert path.is_file()


def test_runtime_yaml_is_removed_when_serialization_fails(tmp_path, monkeypatch):
    train = load_training_module()
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    source = tmp_path / "dataset.yaml"
    source.write_text(yaml.safe_dump({"path": "dataset"}), encoding="utf-8")
    created = []
    original = train.tempfile.NamedTemporaryFile

    def recording_tempfile(*args, **kwargs):
        handle = original(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr(train.tempfile, "NamedTemporaryFile", recording_tempfile)
    monkeypatch.setattr(train.yaml, "safe_dump", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        train.create_runtime_dataset_yaml(source)
    assert created and not created[0].exists()


def test_cleanup_error_does_not_mask_original_exception(tmp_path, monkeypatch):
    train = load_training_module()
    monkeypatch.setattr(train.Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.warns(RuntimeWarning, match="cleanup also failed"):
        train._cleanup_runtime_yaml(tmp_path / "runtime.yaml", ValueError("training failed"))
    with pytest.raises(OSError, match="cleanup failed"):
        train._cleanup_runtime_yaml(tmp_path / "runtime.yaml")


def test_portable_dataset_path():
    config = yaml.safe_load((ROOT / "dataset_p2_cbam.yaml").read_text(encoding="utf-8"))
    assert config["path"] == "data/dataset_yolo_aug_p2_cbam"


def test_train_disables_ultralytics_progress_before_import():
    source = (ROOT / "03_train_p2_cbam.py").read_text(encoding="utf-8")
    quiet_line = 'os.environ.setdefault("YOLO_VERBOSE", "false")'
    import_line = "from ultralytics import YOLO"
    assert quiet_line in source
    assert source.index(quiet_line) < source.index(import_line)


def test_train_cli_overrides_fast_run_controls(tmp_path, monkeypatch):
    train = load_training_module()
    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text("path: dataset\n", encoding="utf-8")
    train_calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            self.model = types.SimpleNamespace(state_dict=lambda: {})

        def train(self, **kwargs):
            train_calls.append(kwargs)

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train, "require_yolo26_support", lambda: object)
    monkeypatch.setattr(train, "register_architecture", lambda architecture: None)
    monkeypatch.setattr(train, "load_partial_pretrained", lambda model, weights: {})
    monkeypatch.setattr(train, "create_runtime_dataset_yaml", lambda source_yaml: runtime_yaml)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sys, "argv", [
        "03_train_p2_cbam.py",
        "--epochs", "2",
        "--fraction", "0.05",
        "--batch", "4",
        "--workers", "0",
        "--name", "quick_smoke",
    ])

    train.main()

    assert train_calls
    assert train_calls[0]["epochs"] == 2
    assert train_calls[0]["fraction"] == pytest.approx(0.05)
    assert train_calls[0]["batch"] == 4
    assert train_calls[0]["workers"] == 0
    assert train_calls[0]["name"] == "quick_smoke"
    assert train_calls[0]["mask_ratio"] == 2
    assert train_calls[0]["optimizer"] == "auto"
    assert train_calls[0]["verbose"] is False
    assert Path(train_calls[0]["project"]) == ROOT / "runs" / "segment"
    assert not runtime_yaml.exists()


def test_train_cli_selects_v2b_model_and_adamw_optimizer(tmp_path, monkeypatch):
    train = load_training_module()
    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text("path: dataset\n", encoding="utf-8")
    train_calls = []
    model_paths = []
    registered = []

    class FakeYOLO:
        def __init__(self, model_path):
            model_paths.append(Path(model_path))
            self.model = types.SimpleNamespace(state_dict=lambda: {})

        def train(self, **kwargs):
            train_calls.append(kwargs)

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train, "require_yolo26_support", lambda: object)
    monkeypatch.setattr(train, "register_architecture", lambda architecture: registered.append(architecture))
    monkeypatch.setattr(train, "load_partial_pretrained", lambda model, weights: {})
    monkeypatch.setattr(train, "create_runtime_dataset_yaml", lambda source_yaml: runtime_yaml)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sys, "argv", [
        "03_train_p2_cbam.py",
        "--architecture", "v2b",
        "--optimizer", "AdamW",
        "--epochs", "1",
        "--batch", "4",
        "--workers", "0",
        "--name", "quick_v2b",
    ])

    train.main()

    assert registered == ["v2b"]
    assert model_paths == [ROOT / "models" / "yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml"]
    assert train_calls[0]["optimizer"] == "AdamW"
    assert train_calls[0]["name"] == "quick_v2b"
    assert not runtime_yaml.exists()
