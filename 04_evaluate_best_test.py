"""Evaluate the locked P2-CBAM best checkpoint on the final test split."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import ultralytics
from ultralytics import YOLO
import yaml

from cbam import P2CompatibleSegment26, register_cbam


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR / "runs" / "segment"
PINNED_ULTRALYTICS_VERSION = "8.4.13"
DEFAULT_RUN_NAME = "SkinSeg_YOLO26_P2_CBAM_Test"
PREFERRED_WEIGHTS = Path("D:/PAPER_SPKT/SkinSeg_YOLO26_P2_CBAM_640/weights/best.pt")
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def resolve_weights_default():
    return PREFERRED_WEIGHTS if PREFERRED_WEIGHTS.is_file() else PREFERRED_WEIGHTS


def default_workers():
    if platform.system() == "Windows":
        return 0
    return 8 if torch.cuda.is_available() else 0


def default_device():
    return 0 if torch.cuda.is_available() else "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate SkinSeg-YOLO26n-P2-CBAM best.pt on test split")
    parser.add_argument("--weights", type=Path, default=resolve_weights_default())
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "dataset_p2_cbam.yaml")
    parser.add_argument("--batch", type=_positive_int, default=16)
    parser.add_argument("--workers", type=_nonnegative_int, default=default_workers())
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _dataset_root(source_yaml, configured_path):
    path = Path(configured_path or ".").expanduser()
    return path.resolve() if path.is_absolute() else (source_yaml.parent / path).resolve()


def resolve_output_dir(name):
    if not name or name in (".", "..") or Path(name).name != name or any(sep in name for sep in ("/", "\\")):
        raise ValueError("--name must be a single directory name with no path separator")
    project = PROJECT_DIR.resolve()
    save_dir = (project / name).resolve()
    if project != save_dir and project not in save_dir.parents:
        raise ValueError("--name escapes the output project directory")
    return save_dir


def lock_path_for(save_dir):
    return save_dir.parent / f".{save_dir.name}.evaluate.lock"


def require_ultralytics_version():
    if ultralytics.__version__ != PINNED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Official evaluation requires ultralytics=={PINNED_ULTRALYTICS_VERSION}; "
            f"found {ultralytics.__version__}."
        )


def validate_inputs(weights, data_yaml):
    weights = Path(weights).expanduser().resolve()
    data_yaml = Path(data_yaml).expanduser().resolve()
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        raise FileNotFoundError(f"Checkpoint .pt file is missing: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML is missing: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    test_value = config.get("test")
    if not isinstance(test_value, str) or not test_value:
        raise ValueError("Dataset YAML must define test as one directory path")
    test_rel = Path(test_value)
    if len(test_rel.parts) == 0:
        raise ValueError("Dataset YAML test path is empty")
    root = _dataset_root(data_yaml, config.get("path", "."))
    image_dir = (root / test_rel).resolve()
    label_parts = list(test_rel.parts)
    if "images" in label_parts:
        label_parts[label_parts.index("images")] = "labels"
        label_dir = root.joinpath(*label_parts).resolve()
    else:
        label_dir = (root / "labels" / test_rel.name).resolve()
    missing = [str(path) for path in (image_dir, label_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Dataset test directories are missing: " + ", ".join(missing))
    return weights, data_yaml


def register_custom_modules():
    register_cbam()


@contextmanager
def exclusive_lock(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, str(os.getpid()).encode("ascii"))
        yield
    except FileExistsError as exc:
        raise FileExistsError(f"Evaluation lock already exists: {lock_path}") from exc
    finally:
        if handle is not None:
            os.close(handle)
            Path(lock_path).unlink(missing_ok=True)


def validate_loaded_model(model):
    inner = getattr(model, "model", model)
    task = getattr(model, "task", getattr(inner, "task", None))
    if task != "segment":
        raise RuntimeError(f"Loaded checkpoint task must be segment, found {task!r}")
    layers = getattr(inner, "model", [])
    head = layers[-1] if layers else None
    if not isinstance(head, P2CompatibleSegment26):
        raise RuntimeError("Loaded checkpoint head is not P2CompatibleSegment26")
    stride_source = getattr(inner, "stride", getattr(head, "stride", None))
    strides = _as_float_list(stride_source)
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Loaded checkpoint stride must be [4, 8, 16, 32], found {strides}")


def _as_float_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def extract_metrics(results_dict):
    metrics = {}
    missing = []
    for key in METRIC_KEYS:
        value = json_scalar((results_dict or {}).get(key))
        metrics[key] = value
        if value is None:
            missing.append(key)
    return metrics, missing


def _args_to_dict(args):
    if args is None:
        return None
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return None


def effective_args(metrics, requested):
    validator = getattr(metrics, "validator", None)
    validator_args = _args_to_dict(getattr(validator, "args", None))
    if validator_args is not None:
        return validator_args, "validator"
    fallback = dict(requested)
    fallback["conf"] = 0.001
    return fallback, "pinned_fallback"


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(str(temp_path), str(path))
    except BaseException:
        try:
            handle.close()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        raise


def build_summary(args, weights, data_yaml, save_dir, val_args, metrics):
    metric_values, missing = extract_metrics(getattr(metrics, "results_dict", {}))
    effective, source = effective_args(metrics, val_args)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(weights),
        "data": str(data_yaml),
        "checkpoint_sha256": sha256_file(weights),
        "split": "test",
        "imgsz": 640,
        "batch": args.batch,
        "workers": args.workers,
        "requested_device": args.device,
        "resolved_device": val_args["device"],
        "seed": args.seed,
        "save_dir": str(save_dir),
        "python_version": sys.version,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "requested_val_args": dict(val_args),
        "effective_val_args": effective,
        "effective_args_source": source,
        "metrics": metric_values,
        "missing_metrics": missing,
    }


def run_evaluation(argv=None):
    args = parse_args(argv)
    if args.device is None:
        args.device = default_device()
    require_ultralytics_version()
    weights, data_yaml = validate_inputs(args.weights, args.data)
    save_dir = resolve_output_dir(args.name)
    if save_dir.exists():
        raise FileExistsError(f"Output directory already exists: {save_dir}")
    lock_path = lock_path_for(save_dir)
    with exclusive_lock(lock_path):
        register_custom_modules()
        model = YOLO(str(weights))
        validate_loaded_model(model)
        val_args = {
            "data": str(data_yaml),
            "split": "test",
            "imgsz": 640,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "plots": True,
            "save_json": True,
            "iou": 0.7,
            "deterministic": True,
            "project": str(PROJECT_DIR.resolve()),
            "name": args.name,
            "exist_ok": False,
            "seed": args.seed,
        }
        metrics = model.val(**val_args)
        returned_save_dir = Path(getattr(metrics, "save_dir")).resolve()
        if returned_save_dir != save_dir:
            raise RuntimeError(f"Ultralytics returned save_dir {returned_save_dir}, expected {save_dir}")
        summary = build_summary(args, weights, data_yaml, save_dir, val_args, metrics)
        output_path = save_dir / "test_metrics.json"
        atomic_write_json(output_path, summary)
        return output_path


def main():
    output_path = run_evaluation()
    print(f"[INFO] Wrote test evaluation summary: {output_path}")


if __name__ == "__main__":
    main()
