import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_test_ci", ROOT / "05_bootstrap_test_ci.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_label(path, class_id, polygon):
    path.write_text(
        " ".join([str(class_id)] + [f"{value:.6f}" for point in polygon for value in point]),
        encoding="utf-8",
    )


def test_bootstrap_ci_writes_summary_outputs_for_single_lesion_dataset(tmp_path):
    module = load_bootstrap_module()
    dataset = tmp_path / "dataset"
    labels = dataset / "labels" / "test"
    images = dataset / "images" / "test"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (images / "img_a.jpg").write_bytes(b"fake")
    (images / "img_b.jpg").write_bytes(b"fake")
    write_label(labels / "img_a.txt", 1, [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    write_label(labels / "img_b.txt", 2, [(0.10, 0.10), (0.40, 0.10), (0.40, 0.40), (0.10, 0.40)])
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(dataset), "test": "images/test", "names": {0: "a", 1: "b", 2: "c"}}),
        encoding="utf-8",
    )
    predictions = [
        {
            "image_id": "img_a",
            "file_name": "img_a.jpg",
            "category_id": 1,
            "score": 0.90,
            "bbox": [16, 16, 32, 32],
            "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
        },
        {
            "image_id": "img_b",
            "file_name": "img_b.jpg",
            "category_id": 1,
            "score": 0.80,
            "bbox": [0, 0, 20, 20],
            "segmentation": [[0, 0, 20, 0, 20, 20, 0, 20]],
        },
    ]
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    metrics_path = tmp_path / "test_metrics.json"
    metrics_path.write_text(json.dumps({"metrics": {"metrics/mAP50(M)": 0.9}}), encoding="utf-8")
    output_dir = tmp_path / "bootstrap"

    summary = module.run_bootstrap_ci(
        data_yaml=data_yaml,
        predictions_json=predictions_path,
        test_metrics_json=metrics_path,
        output_dir=output_dir,
        image_size=64,
        n_boot=200,
        seed=123,
    )

    assert summary["schema_version"] == 1
    assert summary["n_images"] == 2
    assert summary["n_boot"] == 200
    assert summary["ultralytics_point_metrics"]["metrics/mAP50(M)"] == pytest.approx(0.9)
    assert summary["metrics"]["top1_accuracy"]["point"] == pytest.approx(0.5)
    assert summary["metrics"]["mask_iou50_rate"]["point"] == pytest.approx(0.5)
    assert (output_dir / "bootstrap_ci.json").is_file()
    assert (output_dir / "bootstrap_ci.csv").is_file()
    assert (output_dir / "per_image_metrics.csv").is_file()
    assert (output_dir / "bootstrap_ci_table.md").is_file()


def test_prediction_category_offset_is_detected_for_coco_one_based_exports(tmp_path):
    module = load_bootstrap_module()
    dataset = tmp_path / "dataset"
    labels = dataset / "labels" / "test"
    images = dataset / "images" / "test"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (images / "img_a.jpg").write_bytes(b"fake")
    write_label(labels / "img_a.txt", 4, [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(dataset), "test": "images/test", "names": {i: str(i) for i in range(7)}}),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "img_a",
                    "file_name": "img_a.jpg",
                    "category_id": 5,
                    "score": 0.90,
                    "bbox": [16, 16, 32, 32],
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                },
                {
                    "image_id": "other",
                    "file_name": "other.jpg",
                    "category_id": 7,
                    "score": 0.10,
                    "bbox": [0, 0, 1, 1],
                    "segmentation": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = module.per_image_metrics(data_yaml, predictions_path, image_size=64)

    assert rows[0]["true_class"] == 4
    assert rows[0]["pred_class"] == 4
    assert rows[0]["top1_accuracy"] == 1
