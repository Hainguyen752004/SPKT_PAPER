"""
SkinSeg-YOLO26-P2Attn: artifact-oriented fixed multi-view dermoscopic preprocessing.
1. Train v1: source image -> letterbox.
2. Train v7: fixed-threshold/largest-contour crop -> DullRazor-inspired hair
   removal (17x17 cross black-hat, threshold 10, Telea radius 3) -> Gray-World
   -> letterbox.
Validation and test receive v1 letterbox only by default; optional v7 evaluation
views are isolated from the normal split directories.
The fixed v7 chain is applied to every training image; it does not conditionally
detect artifacts. v1 and v7 are legacy identifiers.
"""
import argparse
import os
import cv2
import shutil
import tempfile
import math
import re
import json
import sys
from uuid import uuid4
from collections import Counter
import numpy as np
from tqdm import tqdm
from pathlib import Path

EXPECTED_SOURCE_COUNTS = {"train": 8008, "val": 998, "test": 1007}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "data/dataset_yolo_fixed_labels/dataset_yolo"
DST_ROOT = PROJECT_ROOT / "data/dataset_yolo_640x640_multiview_geom_v2"
PREPROCESSING_VERSION = "paired_geometry_v2"

try:
    from data_processing.paired_geometry import (
        EPSILON, ViewTransform, clip_polygon_to_rect, polygon_area, polygon_intersects_rect,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from paired_geometry import EPSILON, ViewTransform, clip_polygon_to_rect, polygon_area, polygon_intersects_rect


def output_stems(base, split):
    return [f"{base}_v1", f"{base}_v7"] if split == "train" else [f"{base}_v1"]


def resolve_destination(destination, project_root=PROJECT_ROOT, intended_destination=DST_ROOT):
    """Resolve an exact, project-local allow-listed destination without mutation."""
    project_root = Path(project_root).resolve()
    data_path = project_root / "data"
    data_root = data_path.resolve()
    intended = Path(intended_destination)
    if not intended.is_absolute():
        intended = (Path.cwd() / intended).absolute()
    intended = intended.resolve()
    destination = Path(destination).resolve()
    if (data_root.parent != project_root or intended == data_root
            or data_root not in intended.parents or destination != intended):
        raise ValueError("destination is not the exact intended directory below project data")
    return destination


def prepare_destination(destination, project_root, intended_destination, overwrite=False):
    destination = resolve_destination(destination, project_root, intended_destination)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists; pass --overwrite: {destination}")
        shutil.rmtree(destination)
    return destination

def transactional_build(destination, project_root, intended_destination, overwrite, builder):
    destination = resolve_destination(destination, project_root, intended_destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent))
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    while backup.exists():
        backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    backup_created = False
    try:
        builder(temp)
        if destination.exists():
            destination.rename(backup)
            backup_created = True
        try:
            temp.rename(destination)
        except Exception:
            if backup_created and backup.exists() and not destination.exists():
                backup.rename(destination)
                backup_created = False
            raise
        if backup_created and backup.exists():
            shutil.rmtree(backup)
            backup_created = False
    finally:
        if temp.exists(): shutil.rmtree(temp)


def validate_source_layout(source, expected_counts=EXPECTED_SOURCE_COUNTS):
    """Check provenance structure without decoding or parsing its contents."""
    source = Path(source)
    for split, expected in expected_counts.items():
        images = {p.stem for p in (source / "images" / split).glob("*.jpg")}
        labels = {p.stem for p in (source / "labels" / split).glob("*.txt")}
        if images != labels:
            raise ValueError(f"{split}: source image/label stem mismatch")
        if len(images) != expected:
            raise ValueError(f"{split}: unexpected source count {len(images)} != {expected}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def read_image_or_raise(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unreadable input image: {path}")
    return image

# ----------------- THUẬT TOÁN TIỀN XỬ LÝ -----------------
def smart_roi_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0, 0, img.shape[1], img.shape[0]
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w], x, y, w, h

def dullrazor(img, return_mask=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return (inpainted, mask) if return_mask else inpainted

def gray_world_color_constancy(img, return_gains=False):
    b, g, r = cv2.split(img.astype(np.float32))
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        gains = [1.0, 1.0, 1.0]
        corrected = img
    else:
        gains = [float(avg_gray / avg_b), float(avg_gray / avg_g), float(avg_gray / avg_r)]
        b = np.clip(b * gains[0], 0, 255)
        g = np.clip(g * gains[1], 0, 255)
        r = np.clip(r * gains[2], 0, 255)
        corrected = cv2.merge([b, g, r]).astype(np.uint8)
    return (corrected, gains) if return_gains else corrected


def artifact_proxies(mask, crop_width, crop_height, source_width, source_height, gains):
    crop_area = float(crop_width * crop_height)
    source_area = float(source_width * source_height)
    gains = [float(gain) for gain in gains]
    return {
        "hair_mask_coverage": float(np.count_nonzero(mask > 0) / crop_area),
        "vignette_crop_ratio": float(1.0 - crop_area / source_area),
        "gray_world_gains": gains,
        "gray_world_correction_magnitude": float(math.sqrt(sum((gain - 1.0) ** 2 for gain in gains))),
    }

def letterbox_image(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_rs
    return canvas, scale, pad_x, pad_y

# ----------------- HÀM XỬ LÝ NHÃN (LABEL) -----------------
def read_labels(lbl_path):
    items = []
    if not os.path.exists(lbl_path): return items
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) < 7 or (len(parts) - 1) % 2 or not re.fullmatch(r"[+-]?\d+", parts[0]):
                raise ValueError(f"invalid polygon/class in {lbl_path}")
            cls, coords = int(parts[0]), [float(p) for p in parts[1:]]
            if not 0 <= cls <= 6: raise ValueError(f"class outside 0..6 in {lbl_path}")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords): raise ValueError(f"invalid normalized coordinate in {lbl_path}")
            items.append((cls, coords))
    return items

def write_labels(out_path, lines):
    if lines:
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
    else:
        open(out_path, "w").close()

def write_output_pair(image_path, image, label_path, lines):
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"failed to write image: {image_path}")
    try:
        write_labels(label_path, lines)
    except Exception:
        Path(image_path).unlink(missing_ok=True)
        raise

def _polygon_from_coords(coords, source_width, source_height):
    return [[float(coords[index] * source_width), float(coords[index + 1] * source_height)]
            for index in range(0, len(coords), 2)]


def _serialize_polygon_line(class_id, canvas_polygon, canvas_width, canvas_height):
    normalized = []
    for x, y in canvas_polygon:
        normalized.extend([
            float(np.clip(x / canvas_width, 0.0, 1.0)),
            float(np.clip(y / canvas_height, 0.0, 1.0)),
        ])
    return f"{class_id} " + " ".join(f"{coordinate:.6f}" for coordinate in normalized)


def _instance_record(pair_id, source_line_index, class_id, source_polygon,
                     intersection_polygon, canvas_polygon, status, reason):
    return {
        "instance_id": f"{pair_id}:{source_line_index}",
        "class_id": int(class_id),
        "source_polygon": [[float(x), float(y)] for x, y in source_polygon],
        "crop_intersection_polygon": [[float(x), float(y)] for x, y in intersection_polygon],
        "canvas_polygon": [[float(x), float(y)] for x, y in canvas_polygon],
        "source_area": float(polygon_area(source_polygon)),
        "crop_intersection_area": float(polygon_area(intersection_polygon)),
        "canvas_area": float(polygon_area(canvas_polygon)),
        "status": status,
        "reason": reason,
    }


def _transform_items(items, transform, pair_id, clip_to_crop):
    lines = []
    instances = []
    crop_x, crop_y, crop_width, crop_height = transform.crop_box
    for source_line_index, (class_id, coords) in enumerate(items):
        source_polygon = _polygon_from_coords(coords, transform.source_width, transform.source_height)
        if clip_to_crop:
            intersection = clip_polygon_to_rect(
                source_polygon, crop_x, crop_y, crop_x + crop_width, crop_y + crop_height
            )
        else:
            intersection = source_polygon if polygon_area(source_polygon) > 0.0 else []

        if not intersection:
            fully_outside = (
                polygon_area(source_polygon) > EPSILON
                and not polygon_intersects_rect(
                    source_polygon, crop_x, crop_y, crop_x + crop_width, crop_y + crop_height
                )
            )
            reason = "INSTANCE_OUTSIDE_V7_CROP" if fully_outside else "INSTANCE_DEGENERATE_AFTER_CLIP"
            instances.append(_instance_record(
                pair_id, source_line_index, class_id, source_polygon, [], [], "dropped", reason
            ))
            continue

        canvas_polygon = transform.source_to_canvas(intersection)
        if polygon_area(intersection) <= 0.0 or polygon_area(canvas_polygon) <= 0.0:
            instances.append(_instance_record(
                pair_id, source_line_index, class_id, source_polygon, intersection, [],
                "dropped", "INSTANCE_DEGENERATE_AFTER_CLIP"
            ))
            continue
        lines.append(_serialize_polygon_line(
            class_id, canvas_polygon, transform.canvas_width, transform.canvas_height
        ))
        instances.append(_instance_record(
            pair_id, source_line_index, class_id, source_polygon, intersection,
            canvas_polygon, "kept", None
        ))
    return lines, instances


def transform_labels_letterbox(items, old_w, old_h, scale, pad_x, pad_y, new_size=640,
                               pair_id="", return_instances=False):
    transform = ViewTransform(
        source_width=old_w,
        source_height=old_h,
        crop_box=(0, 0, old_w, old_h),
        scale=scale,
        resized_width=int(old_w * scale),
        resized_height=int(old_h * scale),
        pad_x=pad_x,
        pad_y=pad_y,
        canvas_width=new_size,
        canvas_height=new_size,
    )
    lines, instances = _transform_items(items, transform, pair_id, clip_to_crop=False)
    return (lines, instances) if return_instances else lines

def transform_labels_crop_and_letterbox(items, orig_w, orig_h, rx, ry, rw, rh,
                                        scale, pad_x, pad_y, new_size=640, pair_id=""):
    transform = ViewTransform(
        source_width=orig_w,
        source_height=orig_h,
        crop_box=(rx, ry, rw, rh),
        scale=scale,
        resized_width=int(rw * scale),
        resized_height=int(rh * scale),
        pad_x=pad_x,
        pad_y=pad_y,
        canvas_width=new_size,
        canvas_height=new_size,
    )
    return _transform_items(items, transform, pair_id, clip_to_crop=True)

# ----------------- VÒNG LẶP CHÍNH -----------------

        # Góc nhìn 1: Bản gốc

        # Legacy v7: artifact-processed view (fixed crop + hair removal + Gray-World)

def _make_transform(source_width, source_height, crop_box, scale, pad_x, pad_y, canvas_size=640):
    _, _, crop_width, crop_height = crop_box
    return ViewTransform(
        source_width=source_width,
        source_height=source_height,
        crop_box=crop_box,
        scale=scale,
        resized_width=int(crop_width * scale),
        resized_height=int(crop_height * scale),
        pad_x=pad_x,
        pad_y=pad_y,
        canvas_width=canvas_size,
        canvas_height=canvas_size,
    )


def _metadata_record(pair_id, split, view, source, transform, instances, proxies):
    dropped_reasons = Counter(
        instance["reason"] for instance in instances if instance["status"] == "dropped"
    )
    return {
        "schema_version": 1,
        "pair_id": pair_id,
        "split": split,
        "view": view,
        "source": source,
        "transform": transform.to_dict(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "instances": instances,
        "input_instance_count": len(instances),
        "output_instance_count": sum(instance["status"] == "kept" for instance in instances),
        "dropped_instance_reasons": dict(sorted(dropped_reasons.items())),
        "artifact_proxies": proxies,
    }


def process_split(split, dst_root=None, generate_v7_eval=False):
    if dst_root is None:
        dst_root = DST_ROOT
    dst_root = Path(dst_root)
    src_img_dir = Path(SRC_ROOT) / "images" / split
    src_lbl_dir = Path(SRC_ROOT) / "labels" / split
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    img_paths = sorted(src_img_dir.glob("*.jpg"), key=lambda path: path.as_posix())
    records = []
    for img_path in tqdm(img_paths, desc=f"Processing Multi-View {split}"):
        base = img_path.stem
        img = read_image_or_raise(img_path)
        orig_h, orig_w = img.shape[:2]
        items = read_labels(src_lbl_dir / f"{base}.txt")
        source_relative = img_path.relative_to(Path(SRC_ROOT)).as_posix()

        img_v1, s1, px1, py1 = letterbox_image(img.copy(), 640)
        lbl_v1, instances_v1 = transform_labels_letterbox(
            items, orig_w, orig_h, s1, px1, py1, pair_id=base, return_instances=True
        )
        write_output_pair(dst_img_dir / f"{base}_v1.jpg", img_v1,
                          dst_lbl_dir / f"{base}_v1.txt", lbl_v1)
        transform_v1 = _make_transform(orig_w, orig_h, (0, 0, orig_w, orig_h), s1, px1, py1)
        records.append(_metadata_record(
            base, split, "v1", source_relative, transform_v1, instances_v1,
            {
                "hair_mask_coverage": 0.0,
                "vignette_crop_ratio": 0.0,
                "gray_world_gains": [1.0, 1.0, 1.0],
                "gray_world_correction_magnitude": 0.0,
            },
        ))

        if split == "train" or generate_v7_eval:
            img_v7, rx, ry, rw, rh = smart_roi_crop(img.copy())
            img_v7, hair_mask = dullrazor(img_v7, return_mask=True)
            img_v7, gains = gray_world_color_constancy(img_v7, return_gains=True)
            img_v7, s2, px2, py2 = letterbox_image(img_v7, 640)
            lbl_v7, instances_v7 = transform_labels_crop_and_letterbox(
                items, orig_w, orig_h, rx, ry, rw, rh, s2, px2, py2, pair_id=base
            )
            v7_img_dir, v7_lbl_dir = dst_img_dir, dst_lbl_dir
            if split != "train":
                v7_img_dir = dst_root / "images" / f"{split}_v7_eval"
                v7_lbl_dir = dst_root / "labels" / f"{split}_v7_eval"
                ensure_dir(v7_img_dir)
                ensure_dir(v7_lbl_dir)
            write_output_pair(v7_img_dir / f"{base}_v7.jpg", img_v7,
                              v7_lbl_dir / f"{base}_v7.txt", lbl_v7)
            transform_v7 = _make_transform(orig_w, orig_h, (rx, ry, rw, rh), s2, px2, py2)
            records.append(_metadata_record(
                base, split, "v7", source_relative, transform_v7, instances_v7,
                artifact_proxies(hair_mask, rw, rh, orig_w, orig_h, gains),
            ))
    return records


def build_dataset(destination, generate_v7_eval=False):
    """Build images, labels, and deterministic metadata below destination."""
    destination = Path(destination)
    records = []
    for split in ("train", "val", "test"):
        records.extend(process_split(split, destination, generate_v7_eval=generate_v7_eval))
    metadata_dir = destination / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "transforms.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return records


def summarize_output(root):
    images, labels, classes = {}, {}, Counter()
    for split in ("train", "val", "test"):
        images[split] = len(list((Path(root) / "images" / split).glob("*.jpg")))
        paths = list((Path(root) / "labels" / split).glob("*.txt")); labels[split] = len(paths)
        for path in paths:
            for cls, _ in read_labels(path): classes[str(cls)] += 1
    return {"source_pairs": EXPECTED_SOURCE_COUNTS, "output_images": images, "output_labels": labels,
            "invalid_files": [], "skipped_files": 0, "augmentation_attempts": None,
            "augmentation_saved": None, "class_polygons": dict(classes), "cross_split_overlaps": None}

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default=str(DST_ROOT))
    parser.add_argument("--generate-v7-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    destination = resolve_destination(args.destination, PROJECT_ROOT, DST_ROOT)
    print("=" * 50)
    print("SkinSeg-YOLO26-P2Attn: ARTIFACT-ORIENTED FIXED MULTI-VIEW DERMOSCOPIC PREPROCESSING")
    print(f"Nguồn dữ liệu gốc: {SRC_ROOT}")
    print("=" * 50)
    validate_source_layout(SRC_ROOT)
    transactional_build(
        destination, PROJECT_ROOT, DST_ROOT, args.overwrite,
        lambda temp: build_dataset(temp, generate_v7_eval=args.generate_v7_eval),
    )
    summary = summarize_output(destination)
    print(json.dumps(summary, sort_keys=True))
    return summary

if __name__ == "__main__":
    main()
