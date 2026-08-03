"""
SkinSeg-YOLO26-P2Attn: NV-excluding image-level augmentation.
Any image containing class 5 (NV) is skipped; all non-NV images receive up to
three attempts without per-class targets. This stochastic, unseeded stage is
separate from deterministic preprocessing.
"""
import argparse
import warnings
import tempfile
import math
import re
import json
from collections import Counter
import os, glob, cv2, shutil
from pathlib import Path
import numpy as np
import albumentations as A
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Khóa đường dẫn tương thích với 01_preprocess.py (Lưu tại Ham1000_p2_CBAM/data/)
SRC_ROOT = PROJECT_ROOT / "data/dataset_yolo_640x640_multiview"
DST_ROOT = PROJECT_ROOT / "data/dataset_yolo_aug_p2_cbam"

MAJORITY_CLASSES = [5] # NV class
N_AUG_PER_IMAGE = 3
MIN_POLY_AREA_PX = 50

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.25),
    A.Affine(scale=(0.85, 1.15), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-30, 30), p=0.9),
    A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=0.35),
    A.RandomBrightnessContrast(0.12, 0.12, p=0.5),
    A.HueSaturationValue(5, 8, 5, p=0.25),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15)
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def read_image_or_raise(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unreadable input image: {path}")
    return image

def read_yolo_seg(p):
    items = []
    if not os.path.exists(p): return items
    with open(p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) < 7 or (len(parts) - 1) % 2 or not re.fullmatch(r"[+-]?\d+", parts[0]): raise ValueError(f"invalid polygon/class in {p}")
            cls, coords = int(parts[0]), list(map(float, parts[1:]))
            if not 0 <= cls <= 6: raise ValueError(f"class outside 0..6 in {p}")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords): raise ValueError(f"invalid normalized coordinate in {p}")
            items.append((cls, list(zip(coords[::2], coords[1::2]))))
    return items

def poly_area(poly_px):
    x = np.array([p[0] for p in poly_px], dtype=np.float32)
    y = np.array([p[1] for p in poly_px], dtype=np.float32)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def should_augment(items):
    return not any(item[0] in MAJORITY_CLASSES for item in items)


def run_augmentation_attempts(attempt):
    saved = 0
    for index in range(N_AUG_PER_IMAGE):
        saved += bool(attempt(index))
    return {"augmentation_attempts": N_AUG_PER_IMAGE, "augmentation_saved": saved,
            "skipped_files": N_AUG_PER_IMAGE - saved}


def augment_attempt(img, items, w, h, base, k, dst_img, dst_lbl):
    keypoints, kp_map = [], []
    for item_idx, (_, poly) in enumerate(items):
        for point in poly:
            keypoints.append((point[0] * w, point[1] * h)); kp_map.append(item_idx)
    if not keypoints:
        return False
    try:
        augmented = transform(image=img, keypoints=keypoints)
    except (ValueError, cv2.error) as exc:
        warnings.warn(f"augmentation {base} attempt {k} skipped: {exc}", RuntimeWarning)
        return False
    grouped = {i: [] for i in range(len(items))}
    for point, item_idx in zip(augmented["keypoints"], kp_map): grouped[item_idx].append(point)
    lines = []
    for item_idx, (class_id, _) in enumerate(items):
        points = grouped[item_idx]
        if len(points) < 3 or poly_area(np.asarray(points, dtype=np.float32)) < MIN_POLY_AREA_PX: continue
        coords = []
        for x, y in points: coords.extend([np.clip(x / w, 0, 1), np.clip(y / h, 0, 1)])
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))
    if not lines: return False
    out_base = f"{base}_aug{k}"
    image_path = os.path.join(dst_img, out_base + ".jpg")
    label_path = os.path.join(dst_lbl, out_base + ".txt")
    if not cv2.imwrite(image_path, augmented["image"]):
        raise OSError(f"failed to write image: {image_path}")
    try:
        with open(label_path, "w") as handle: handle.write("\n".join(lines))
    except Exception:
        Path(image_path).unlink(missing_ok=True)
        raise
    return True


def prepare_destination(destination, project_root, intended_destination, overwrite=False):
    destination, project_root = Path(destination).resolve(), Path(project_root).resolve()
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
    data_root = Path(project_root).resolve() / "data"
    if destination != intended or destination == data_root or data_root not in destination.parents: raise ValueError("destination is not the exact intended directory below project data")
    if destination.exists() and not overwrite: raise FileExistsError(f"destination exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent)); backup = destination.with_name(f".{destination.name}.backup")
    try:
        builder(temp)
        if destination.exists(): destination.rename(backup)
        try: temp.rename(destination)
        except Exception:
            if backup.exists() and not destination.exists(): backup.rename(destination)
            raise
        if backup.exists(): shutil.rmtree(backup)
    finally:
        if temp.exists(): shutil.rmtree(temp)


def validate_source_layout(source):
    source = Path(source)
    for split in ("train", "val", "test"):
        image_dir, label_dir = source / "images" / split, source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Stage 2 source layout missing {split}: {source}")
        image_stems = {p.stem for p in image_dir.glob("*.jpg")}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        if not image_stems or not label_stems:
            raise ValueError(f"Stage 2 source split is empty in {split}: {source}")
        if image_stems != label_stems:
            raise ValueError(f"Stage 2 source stem mismatch in {split}: {source}")


def prepare_pipeline_destination(source, destination, project_root, intended_destination, overwrite=False):
    validate_source_layout(source)
    return prepare_destination(destination, project_root, intended_destination, overwrite)

def augment_train(dst_root=None):
    if dst_root is None:
        dst_root = DST_ROOT
    src_img = os.path.join(SRC_ROOT, "images", "train")
    src_lbl = os.path.join(SRC_ROOT, "labels", "train")
    dst_img = os.path.join(dst_root, "images", "train")
    dst_lbl = os.path.join(dst_root, "labels", "train")
    
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    
    summary = {"augmentation_attempts": 0, "augmentation_saved": 0, "skipped_files": 0}
    img_files = glob.glob(os.path.join(src_img, "*.jpg"))
    for img_path in tqdm(img_files, desc="Augmenting P2-CBAM Dataset"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_lbl, base + ".txt")
        img = read_image_or_raise(img_path)
        h, w = img.shape[:2]
        items = read_yolo_seg(lbl_path)
        
        # Sao chép bản gốc
        shutil.copy(img_path, os.path.join(dst_img, base + ".jpg"))
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dst_lbl, base + ".txt"))
            
        if not should_augment(items):
            continue
            
        # Augment every eligible non-NV image equally; no per-class target frequency.
        item_summary = run_augmentation_attempts(lambda k: augment_attempt(img, items, w, h, base, k, dst_img, dst_lbl))
        for key in summary: summary[key] += item_summary[key]
    return summary

def summarize_output(root, augmentation):
    images, labels, classes = {}, {}, Counter()
    for split in ("train", "val", "test"):
        images[split] = len(list((Path(root) / "images" / split).glob("*.jpg")))
        label_paths = list((Path(root) / "labels" / split).glob("*.txt")); labels[split] = len(label_paths)
        for path in label_paths:
            for cls, _ in read_yolo_seg(path): classes[str(cls)] += 1
    return {"source_pairs": None, "output_images": images, "output_labels": labels,
            "invalid_files": [], "skipped_files": augmentation["skipped_files"],
            "augmentation_attempts": augmentation["augmentation_attempts"],
            "augmentation_saved": augmentation["augmentation_saved"],
            "class_polygons": dict(classes), "cross_split_overlaps": None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print("=" * 50)
    print("SkinSeg-YOLO26-P2Attn: NV-EXCLUDING IMAGE-LEVEL AUGMENTATION")
    print(f"Nguồn dữ liệu vào: {SRC_ROOT}")
    print(f"Đích xuất dữ liệu: {DST_ROOT}")
    print("=" * 50)
    validate_source_layout(SRC_ROOT)
    result = {}
    def build(temp):
        result.update(augment_train(temp))
        for split in ("val", "test"):
            shutil.copytree(Path(SRC_ROOT) / "images" / split, temp / "images" / split)
            shutil.copytree(Path(SRC_ROOT) / "labels" / split, temp / "labels" / split)
    transactional_build(DST_ROOT, PROJECT_ROOT, PROJECT_ROOT / "data/dataset_yolo_aug_p2_cbam",
                        args.overwrite, build)
    summary = summarize_output(DST_ROOT, result)
    print(json.dumps(summary, sort_keys=True))
    return summary

if __name__ == "__main__":
    main()
