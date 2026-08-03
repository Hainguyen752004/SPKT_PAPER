"""
SkinSeg-YOLO26-P2Attn: artifact-oriented fixed multi-view dermoscopic preprocessing.
1. Train v1: source image -> letterbox.
2. Train v7: fixed-threshold/largest-contour crop -> DullRazor-inspired hair
   removal (17x17 cross black-hat, threshold 10, Telea radius 3) -> Gray-World
   -> letterbox.
Validation and test receive v1 letterbox only; no v7 view is generated.
The fixed v7 chain is applied to every training image; it does not conditionally
detect artifacts. v1 and v7 are legacy identifiers.
"""
import argparse
import os
import glob
import cv2
import shutil
import tempfile
import math
import re
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
from pathlib import Path

EXPECTED_SOURCE_COUNTS = {"train": 8008, "val": 998, "test": 1007}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "data/dataset_yolo_fixed_labels/dataset_yolo"
DST_ROOT = PROJECT_ROOT / "data/dataset_yolo_640x640_multiview"


def output_stems(base, split):
    return [f"{base}_v1", f"{base}_v7"] if split == "train" else [f"{base}_v1"]


def prepare_destination(destination, project_root, intended_destination, overwrite=False):
    destination = Path(destination).resolve()
    project_root = Path(project_root).resolve()
    intended = Path(intended_destination).resolve()
    data_root = (project_root / "data").resolve()
    if destination != intended or destination == data_root or data_root not in destination.parents:
        raise ValueError("destination is not the exact intended directory below project data")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists; pass --overwrite: {destination}")
        shutil.rmtree(destination)
    return destination

def transactional_build(destination, project_root, intended_destination, overwrite, builder):
    destination, intended = Path(destination).resolve(), Path(intended_destination).resolve()
    data_root = (Path(project_root).resolve() / "data")
    if destination != intended or destination == data_root or data_root not in destination.parents:
        raise ValueError("destination is not the exact intended directory below project data")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent))
    backup = destination.with_name(f".{destination.name}.backup")
    try:
        builder(temp)
        if destination.exists(): destination.rename(backup)
        try:
            temp.rename(destination)
        except Exception:
            if backup.exists() and not destination.exists(): backup.rename(destination)
            raise
        if backup.exists(): shutil.rmtree(backup)
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

def dullrazor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def gray_world_color_constancy(img):
    b, g, r = cv2.split(img.astype(np.float32))
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return img
    b = np.clip(b * (avg_gray / avg_b), 0, 255)
    g = np.clip(g * (avg_gray / avg_g), 0, 255)
    r = np.clip(r * (avg_gray / avg_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

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

def transform_labels_letterbox(items, old_w, old_h, scale, pad_x, pad_y, new_size=640):
    lines = []
    for cls, coords in items:
        new_coords = []
        for i in range(0, len(coords), 2):
            x = (coords[i] * old_w) * scale + pad_x
            y = (coords[i+1] * old_h) * scale + pad_y
            x = np.clip(x / new_size, 0.0, 1.0)
            y = np.clip(y / new_size, 0.0, 1.0)
            new_coords.extend([x, y])
        lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in new_coords))
    return lines

def transform_labels_crop_and_letterbox(items, orig_w, orig_h, rx, ry, rw, rh, scale, pad_x, pad_y, new_size=640):
    lines = []
    for cls, coords in items:
        new_coords = []
        for i in range(0, len(coords), 2):
            px = coords[i] * orig_w
            py = coords[i+1] * orig_h
            px = px - rx
            py = py - ry
            px = np.clip(px, 0, rw)
            py = np.clip(py, 0, rh)
            px = px * scale + pad_x
            py = py * scale + pad_y
            px_norm = np.clip(px / new_size, 0.0, 1.0)
            py_norm = np.clip(py / new_size, 0.0, 1.0)
            new_coords.extend([px_norm, py_norm])
        lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in new_coords))
    return lines

# ----------------- VÒNG LẶP CHÍNH -----------------
def process_split(split, dst_root=None):
    if dst_root is None:
        dst_root = DST_ROOT
    src_img_dir = os.path.join(SRC_ROOT, "images", split)
    src_lbl_dir = os.path.join(SRC_ROOT, "labels", split)
    dst_img_dir = os.path.join(dst_root, "images", split)
    dst_lbl_dir = os.path.join(dst_root, "labels", split)
    
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)
    
    img_paths = glob.glob(os.path.join(src_img_dir, "*.jpg"))
    if not img_paths: return

    for img_path in tqdm(img_paths, desc=f"Processing Multi-View {split}"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_lbl_dir, base + ".txt")
        img = read_image_or_raise(img_path)
        orig_h, orig_w = img.shape[:2]
        items = read_labels(lbl_path)

        # Góc nhìn 1: Bản gốc
        img_v1, s1, px1, py1 = letterbox_image(img.copy(), 640)
        lbl_v1 = transform_labels_letterbox(items, orig_w, orig_h, s1, px1, py1)
        write_output_pair(os.path.join(dst_img_dir, f"{base}_v1.jpg"), img_v1,
                          os.path.join(dst_lbl_dir, f"{base}_v1.txt"), lbl_v1)

        # Legacy v7: artifact-processed view (fixed crop + hair removal + Gray-World)
        if split == "train":
            img_v7, rx, ry, rw, rh = smart_roi_crop(img.copy())
            img_v7 = dullrazor(img_v7)
            img_v7 = gray_world_color_constancy(img_v7)
            img_v7, s2, px2, py2 = letterbox_image(img_v7, 640)
            lbl_v7 = transform_labels_crop_and_letterbox(items, orig_w, orig_h, rx, ry, rw, rh, s2, px2, py2)
            write_output_pair(os.path.join(dst_img_dir, f"{base}_v7.jpg"), img_v7,
                              os.path.join(dst_lbl_dir, f"{base}_v7.txt"), lbl_v7)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print("=" * 50)
    print("SkinSeg-YOLO26-P2Attn: ARTIFACT-ORIENTED FIXED MULTI-VIEW DERMOSCOPIC PREPROCESSING")
    print(f"Nguồn dữ liệu gốc: {SRC_ROOT}")
    print("=" * 50)
    validate_source_layout(SRC_ROOT)
    transactional_build(DST_ROOT, PROJECT_ROOT, PROJECT_ROOT / "data/dataset_yolo_640x640_multiview",
                        args.overwrite, lambda temp: [process_split(split, temp) for split in ("train", "val", "test")])
    summary = summarize_output(DST_ROOT)
    print(json.dumps(summary, sort_keys=True))
    return summary

if __name__ == "__main__":
    main()
