"""Bootstrap image-level confidence intervals for a saved YOLO test run.

This script complements Ultralytics mAP point estimates with image-level
bootstrap intervals for interpretable proxy metrics. It does not recompute
COCO mAP; official mAP values are copied from test_metrics.json as point
estimates and should be reported separately.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR.parent / "SkinSeg_YOLO26_P2_CBAM_640_final" / "SkinSeg_YOLO26_P2_CBAM_Test_Final_20260817"
DEFAULT_DATA_YAML = SCRIPT_DIR / "dataset_p2_cbam.yaml"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "bootstrap_ci"
METRIC_COLUMNS = [
    "top1_accuracy",
    "box_iou_mean",
    "box_iou50_rate",
    "mask_iou_mean",
    "mask_dice_mean",
    "mask_iou50_rate",
    "strict_class_and_mask_iou50_rate",
]
MACRO_METRIC_COLUMNS = [
    "macro_top1_accuracy",
    "macro_mask_iou_mean",
    "macro_mask_dice_mean",
    "macro_strict_class_and_mask_iou50_rate",
]


def _dataset_root(source_yaml, configured_path):
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(source_yaml).resolve().parent / path).resolve()


def load_dataset_paths(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = _dataset_root(data_yaml, config.get("path", "."))
    test_images = root / config.get("test", "images/test")
    test_labels = root / Path(config.get("test", "images/test")).as_posix().replace("images/", "labels/", 1)
    if not test_images.is_dir():
        raise FileNotFoundError(f"Missing test image directory: {test_images}")
    if not test_labels.is_dir():
        raise FileNotFoundError(f"Missing test label directory: {test_labels}")
    return test_images, test_labels, config.get("names", {})


def load_predictions(predictions_json, class_offset=0):
    predictions = json.loads(Path(predictions_json).read_text(encoding="utf-8"))
    by_image = {}
    for pred in predictions:
        pred = dict(pred)
        pred["category_id"] = int(pred["category_id"]) - class_offset
        key = Path(pred.get("file_name") or f"{pred['image_id']}.jpg").stem
        by_image.setdefault(key, []).append(pred)
    return by_image


def infer_prediction_class_offset(predictions_json, class_names):
    """Detect COCO one-based category IDs exported by Ultralytics save_json."""
    predictions = json.loads(Path(predictions_json).read_text(encoding="utf-8"))
    if not predictions or not class_names:
        return 0
    category_ids = [int(pred["category_id"]) for pred in predictions if "category_id" in pred]
    if not category_ids:
        return 0
    class_count = len(class_names)
    if min(category_ids) >= 1 and max(category_ids) == class_count:
        return 1
    return 0


def read_label(label_path, image_size):
    lines = [line.strip() for line in Path(label_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty label file: {label_path}")
    parts = lines[0].split()
    class_id = int(float(parts[0]))
    coords = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
    pixels = coords * float(image_size)
    x_min, y_min = pixels.min(axis=0)
    x_max, y_max = pixels.max(axis=0)
    mask = polygon_to_mask(pixels.reshape(-1).tolist(), image_size)
    return {"class_id": class_id, "bbox": [x_min, y_min, x_max - x_min, y_max - y_min], "mask": mask}


def polygon_to_mask(polygon, image_size):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    if not polygon:
        return mask.astype(bool)
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points = np.rint(points).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, image_size - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_size - 1)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _compressed_rle_counts(counts):
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    values = []
    index = 0
    while index < len(counts):
        value = 0
        shift = 0
        while True:
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << shift
            shift += 5
            if not (char_value & 0x20):
                if char_value & 0x10:
                    value |= -1 << shift
                break
        if len(values) > 2:
            value += values[-2]
        values.append(value)
    return values


def rle_to_mask(rle):
    height, width = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = _compressed_rle_counts(counts)
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    value = 0
    for run_length in counts:
        if run_length < 0:
            raise ValueError("Invalid negative RLE run length")
        if value == 1:
            flat[cursor: cursor + run_length] = 1
        cursor += run_length
        value = 1 - value
    return flat.reshape((width, height)).T.astype(bool)


def prediction_mask(segmentation, image_size):
    if segmentation is None:
        return np.zeros((image_size, image_size), dtype=bool)
    if isinstance(segmentation, dict):
        mask = rle_to_mask(segmentation)
        if mask.shape != (image_size, image_size):
            mask = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask
    polygons = segmentation if isinstance(segmentation, list) else []
    mask = np.zeros((image_size, image_size), dtype=bool)
    for polygon in polygons:
        mask |= polygon_to_mask(polygon, image_size)
    return mask


def box_iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def mask_iou_and_dice(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    union = area_a + area_b - intersection
    iou = 0.0 if union == 0 else float(intersection / union)
    dice_denominator = area_a + area_b
    dice = 0.0 if dice_denominator == 0 else float(2 * intersection / dice_denominator)
    return iou, dice


def per_image_metrics(data_yaml, predictions_json, image_size):
    _, labels_dir, class_names = load_dataset_paths(data_yaml)
    class_offset = infer_prediction_class_offset(predictions_json, class_names)
    predictions = load_predictions(predictions_json, class_offset=class_offset)
    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_id = label_path.stem
        truth = read_label(label_path, image_size)
        candidates = predictions.get(image_id, [])
        top_prediction = max(candidates, key=lambda item: float(item.get("score", 0.0)), default=None)
        if top_prediction is None:
            pred_class = -1
            score = 0.0
            box_iou = 0.0
            mask_iou = 0.0
            mask_dice = 0.0
        else:
            pred_class = int(top_prediction["category_id"])
            score = float(top_prediction.get("score", 0.0))
            box_iou = box_iou_xywh(truth["bbox"], top_prediction.get("bbox", [0, 0, 0, 0]))
            pred_mask = prediction_mask(top_prediction.get("segmentation"), image_size)
            mask_iou, mask_dice = mask_iou_and_dice(truth["mask"], pred_mask)
        class_correct = int(pred_class == truth["class_id"])
        rows.append(
            {
                "image_id": image_id,
                "true_class": truth["class_id"],
                "pred_class": pred_class,
                "score": score,
                "top1_accuracy": class_correct,
                "box_iou": box_iou,
                "box_iou50": int(box_iou >= 0.50),
                "mask_iou": mask_iou,
                "mask_dice": mask_dice,
                "mask_iou50": int(mask_iou >= 0.50),
                "strict_class_and_mask_iou50": int(class_correct and mask_iou >= 0.50),
            }
        )
    if not rows:
        raise ValueError(f"No label files found for test split in {labels_dir}")
    return rows


def summarize_rows(rows):
    arrays = _metric_arrays(rows)
    return {name: float(values.mean()) for name, values in arrays.items()}


def _metric_arrays(rows):
    return {
        "top1_accuracy": np.array([row["top1_accuracy"] for row in rows], dtype=float),
        "box_iou_mean": np.array([row["box_iou"] for row in rows], dtype=float),
        "box_iou50_rate": np.array([row["box_iou50"] for row in rows], dtype=float),
        "mask_iou_mean": np.array([row["mask_iou"] for row in rows], dtype=float),
        "mask_dice_mean": np.array([row["mask_dice"] for row in rows], dtype=float),
        "mask_iou50_rate": np.array([row["mask_iou50"] for row in rows], dtype=float),
        "strict_class_and_mask_iou50_rate": np.array(
            [row["strict_class_and_mask_iou50"] for row in rows], dtype=float
        ),
    }


def bootstrap_summary(rows, n_boot, seed, ci_level=0.95):
    rng = np.random.default_rng(seed)
    arrays = _metric_arrays(rows)
    n = len(rows)
    alpha = (1.0 - ci_level) / 2.0
    lower_q, upper_q = alpha * 100.0, (1.0 - alpha) * 100.0
    summary = {}
    for name, values in arrays.items():
        samples = np.empty(n_boot, dtype=float)
        for index in range(n_boot):
            indices = rng.integers(0, n, size=n)
            samples[index] = values[indices].mean()
        summary[name] = {
            "point": float(values.mean()),
            "ci_low": float(np.percentile(samples, lower_q)),
            "ci_high": float(np.percentile(samples, upper_q)),
        }
    summary.update(stratified_macro_bootstrap_summary(rows, n_boot, rng, lower_q, upper_q))
    return summary


def per_class_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(int(row["true_class"]), []).append(row)
    summaries = []
    for class_id, group in sorted(groups.items()):
        summaries.append(
            {
                "class_id": class_id,
                "n": len(group),
                "top1_accuracy": float(np.mean([row["top1_accuracy"] for row in group])),
                "mask_iou_mean": float(np.mean([row["mask_iou"] for row in group])),
                "mask_dice_mean": float(np.mean([row["mask_dice"] for row in group])),
                "strict_class_and_mask_iou50_rate": float(
                    np.mean([row["strict_class_and_mask_iou50"] for row in group])
                ),
            }
        )
    return summaries


def macro_points(rows):
    classes = per_class_summary(rows)
    return {
        "macro_top1_accuracy": float(np.mean([row["top1_accuracy"] for row in classes])),
        "macro_mask_iou_mean": float(np.mean([row["mask_iou_mean"] for row in classes])),
        "macro_mask_dice_mean": float(np.mean([row["mask_dice_mean"] for row in classes])),
        "macro_strict_class_and_mask_iou50_rate": float(
            np.mean([row["strict_class_and_mask_iou50_rate"] for row in classes])
        ),
    }


def stratified_macro_bootstrap_summary(rows, n_boot, rng, lower_q, upper_q):
    groups = {}
    for row in rows:
        groups.setdefault(int(row["true_class"]), []).append(row)
    point = macro_points(rows)
    samples = {name: np.empty(n_boot, dtype=float) for name in MACRO_METRIC_COLUMNS}
    for index in range(n_boot):
        sampled_rows = []
        for group in groups.values():
            indices = rng.integers(0, len(group), size=len(group))
            sampled_rows.extend(group[int(i)] for i in indices)
        sampled_points = macro_points(sampled_rows)
        for name in MACRO_METRIC_COLUMNS:
            samples[name][index] = sampled_points[name]
    return {
        name: {
            "point": point[name],
            "ci_low": float(np.percentile(values, lower_q)),
            "ci_high": float(np.percentile(values, upper_q)),
        }
        for name, values in samples.items()
    }


def load_point_metrics(test_metrics_json):
    if test_metrics_json is None:
        return {}
    path = Path(test_metrics_json)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", {})


def write_outputs(output_dir, rows, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "point", "ci_low", "ci_high"])
        writer.writeheader()
        for metric, values in summary["metrics"].items():
            writer.writerow({"metric": metric, **values})
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    per_class_rows = per_class_summary(rows)
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(per_class_rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)
    lines = [
        "| Metric | Point | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for metric, values in summary["metrics"].items():
        lines.append(
            f"| {metric} | {values['point']:.4f} | [{values['ci_low']:.4f}, {values['ci_high']:.4f}] |"
        )
    (output_dir / "bootstrap_ci_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bootstrap_ci(data_yaml, predictions_json, test_metrics_json, output_dir, image_size, n_boot, seed):
    rows = per_image_metrics(data_yaml, predictions_json, image_size)
    summary = {
        "schema_version": 1,
        "n_images": len(rows),
        "n_boot": n_boot,
        "seed": seed,
        "image_size": image_size,
        "ci_method": "image-level nonparametric bootstrap percentile CI",
        "metric_scope": "image-level proxy metrics; Ultralytics mAP values are point estimates copied from test_metrics_json",
        "ultralytics_point_metrics": load_point_metrics(test_metrics_json),
        "metrics": bootstrap_summary(rows, n_boot, seed),
    }
    write_outputs(output_dir, rows, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap image-level confidence intervals for saved test predictions")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_RUN_DIR / "predictions.json")
    parser.add_argument("--test-metrics", type=Path, default=DEFAULT_RUN_DIR / "test_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_bootstrap_ci(
        data_yaml=args.data,
        predictions_json=args.predictions,
        test_metrics_json=args.test_metrics,
        output_dir=args.output_dir,
        image_size=args.imgsz,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"[INFO] Wrote bootstrap CI outputs to: {Path(args.output_dir).resolve()}")
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['point']:.4f} [{values['ci_low']:.4f}, {values['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
