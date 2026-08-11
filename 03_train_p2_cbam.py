"""Train the SkinSeg-YOLO26n P2-CBAM segmentation model."""
import argparse
import os
import platform
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import yaml
from ultralytics import YOLO

from cbam import CBAM, register_cbam


SCRIPT_DIR = Path(__file__).resolve().parent
register_cbam()

# Stock YOLO26-seg layer -> custom P2-CBAM layer. Only semantically unchanged
# backbone and top-down layers are transferable; the rebuilt lower head is excluded.
YOLO26_SEG_LAYER_MAP = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12,
    11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18,
}


def pretrained_weights_path():
    """Return the packaged official YOLO26n-seg checkpoint path."""
    return SCRIPT_DIR / "models" / "yolo26n-seg.pt"


def require_yolo26_support():
    """Fail before parsing when the installed Ultralytics predates YOLO26."""
    try:
        from ultralytics.nn.modules import Segment26
    except ImportError as exc:
        raise RuntimeError(
            "Installed Ultralytics is incompatible: Segment26 is unavailable. "
            "Install Ultralytics 8.4.60 (the version targeted by this project)."
        ) from exc
    if Segment26 is None:
        raise RuntimeError("Ultralytics Segment26 is unavailable; install Ultralytics 8.4.60.")
    return Segment26


def _dataset_root(source_yaml, configured_path):
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(source_yaml).resolve().parent / path).resolve()


def create_runtime_dataset_yaml(source_yaml):
    """Validate all splits and write a collision-safe YAML without mutating the source."""
    source_yaml = Path(source_yaml).resolve()
    with source_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = _dataset_root(source_yaml, config.get("path", "."))
    missing = []
    for split in ("train", "val", "test"):
        image_relative = Path(config.get(split, f"images/{split}"))
        image_dir = root / image_relative
        label_parts = list(image_relative.parts)
        if "images" in label_parts:
            label_parts[label_parts.index("images")] = "labels"
            label_dir = root.joinpath(*label_parts)
        else:
            label_dir = root / "labels" / split
        for directory in (image_dir, label_dir):
            if not directory.is_dir():
                missing.append(str(directory))
    if missing:
        raise FileNotFoundError("Dataset directories are missing: " + ", ".join(missing))
    runtime = dict(config)
    runtime["path"] = root.as_posix()
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="yolo26_dataset_", encoding="utf-8", delete=False
    )
    runtime_path = Path(handle.name)
    try:
        yaml.safe_dump(runtime, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        runtime_path.unlink(missing_ok=True)
        raise
    else:
        handle.close()
    return runtime_path


def _remap_state_key(source_key):
    parts = source_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    destination_index = YOLO26_SEG_LAYER_MAP.get(int(parts[1]))
    return None if destination_index is None else f"model.{destination_index}.{parts[2]}"


def build_transfer_report(destination_state, source_state):
    """Remap only verified semantic layers, then classify shape-compatible tensors."""
    matched = {}
    mapped = []
    shape_mismatched = []
    skipped_unmapped = []
    considered_destinations = set()
    for source_key, source_tensor in source_state.items():
        destination_key = _remap_state_key(source_key)
        if destination_key is None:
            skipped_unmapped.append(source_key)
            continue
        considered_destinations.add(destination_key)
        destination_tensor = destination_state.get(destination_key)
        if destination_tensor is None:
            continue
        if source_tensor.shape == destination_tensor.shape:
            matched[destination_key] = source_tensor
            mapped.append({"source": source_key, "destination": destination_key})
        else:
            shape_mismatched.append({"source": source_key, "destination": destination_key})
    transferable_destination_keys = {
        key for key in destination_state
        if len(key.split(".", 2)) == 3 and key.split(".", 2)[1].isdigit()
        and int(key.split(".", 2)[1]) in set(YOLO26_SEG_LAYER_MAP.values())
    }
    report = {
        "mapped": sorted(mapped, key=lambda item: item["destination"]),
        "missing_destination": sorted(transferable_destination_keys - considered_destinations),
        "shape_mismatched": sorted(shape_mismatched, key=lambda item: item["destination"]),
        "skipped_unmapped": sorted(skipped_unmapped),
    }
    return report, matched


def load_partial_pretrained(model, weights_path):
    """Transfer every shape-compatible official YOLO26n-seg tensor."""
    weights_path = Path(weights_path)
    empty = {"mapped": [], "missing_destination": [], "shape_mismatched": [], "skipped_unmapped": []}
    if not weights_path.is_file():
        warnings.warn(f"Pretrained weights not found at {weights_path}; initializing from YAML.", UserWarning)
        return empty
    source_model = YOLO(str(weights_path)).model
    destination_model = model.model
    report, matched = build_transfer_report(destination_model.state_dict(), source_model.state_dict())
    destination_model.load_state_dict(matched, strict=False)
    print(
        "[TRANSFER] matched={matched} missing_destination={missing} shape_mismatched={shapes}".format(
            matched=len(report["mapped"]),
            missing=len(report["missing_destination"]),
            shapes=len(report["shape_mismatched"]),
        )
    )
    return report


def _cleanup_runtime_yaml(path, original_exception=None):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as cleanup_error:
        if original_exception is not None:
            warnings.warn(f"Training failed and runtime YAML cleanup also failed: {cleanup_error}", RuntimeWarning)
        else:
            raise


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


def _fraction(value):
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Train SkinSeg-YOLO26n-P2-CBAM")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    parser.add_argument("--fraction", type=_fraction, default=1.0, help="Dataset fraction for quick checks (default: 1.0)")
    parser.add_argument("--batch", type=_positive_int, default=None, help="Override batch size")
    parser.add_argument("--workers", type=_nonnegative_int, default=None, help="Override dataloader workers")
    parser.add_argument("--name", default=None, help="Override Ultralytics run name")
    args = parser.parse_args()
    require_yolo26_support()
    model_yaml = SCRIPT_DIR / "models" / "yolo26n-seg-p2-cbam.yaml"
    weights = pretrained_weights_path()
    dataset_yaml = SCRIPT_DIR / "dataset_p2_cbam.yaml"
    runtime_yaml = create_runtime_dataset_yaml(dataset_yaml)
    original_exception = None
    try:
        print(f"[INFO] SkinSeg-YOLO26n P2-CBAM model: {model_yaml}")
        model = YOLO(str(model_yaml))
        load_partial_pretrained(model, weights)
        device = 0 if torch.cuda.is_available() else "cpu"
        is_windows = platform.system() == "Windows"
        workers = args.workers if args.workers is not None else (0 if is_windows else (8 if device == 0 else 0))
        batch_size = args.batch if args.batch is not None else (8 if is_windows else (32 if device == 0 else 8))
        run_name = args.name or (
            "SkinSeg_YOLO26_P2_CBAM_Conference" if args.epochs > 20 else "SkinSeg_YOLO26_P2_CBAM_Test"
        )
        model.train(
            data=str(runtime_yaml), epochs=args.epochs, batch=batch_size, imgsz=640,
            device=device, workers=workers, cos_lr=True,
            close_mosaic=min(10, max(1, args.epochs // 5)),
            warmup_epochs=3.0 if args.epochs >= 20 else 1.0,
            mosaic=1.0, mixup=0.1, copy_paste=0.1, flipud=0.5,
            degrees=180.0, scale=0.5,
            project=str(SCRIPT_DIR / "runs" / "segment"),
            name=run_name,
            exist_ok=True, val=True, fraction=args.fraction, mask_ratio=2,
            verbose=False,
        )
    except BaseException as exc:
        original_exception = exc
        raise
    finally:
        _cleanup_runtime_yaml(runtime_yaml, original_exception)


if __name__ == "__main__":
    main()
