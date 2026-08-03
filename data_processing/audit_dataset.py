"""Read-only validation, leakage audit, and preservation manifests for datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

SPLITS = ("train", "val", "test")
EXPECTED_SOURCE_COUNTS = {"train": 8008, "val": 998, "test": 1007}
SOURCE_RELATIVE = Path("data/dataset_yolo_fixed_labels/dataset_yolo")
OUTPUT_RELATIVES = (
    Path("data/dataset_yolo_640x640_multiview"),
    Path("data/dataset_yolo_aug_p2_cbam"),
)


def parse_yolo_lines(lines, path=Path("<label>")):
    items = []
    for number, raw in enumerate(lines, 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(f"{path}:{number}: expected class and at least 3 coordinate pairs")
        if not re.fullmatch(r"[+-]?\d+", parts[0]):
            raise ValueError(f"{path}:{number}: class must be an integer")
        class_id = int(parts[0])
        if not 0 <= class_id <= 6:
            raise ValueError(f"{path}:{number}: class outside 0..6")
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: non-numeric coordinate") from exc
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
            raise ValueError(f"{path}:{number}: coordinates must be finite and normalized")
        items.append((class_id, list(zip(coords[::2], coords[1::2]))))
    return items


def normalized_source_id(filename):
    stem = Path(filename).stem.casefold()
    while True:
        previous = stem
        stem = re.sub(r"_aug\d+$", "", stem)
        stem = re.sub(r"_(?:v1|v7)$", "", stem)
        if stem == previous:
            break
    if not re.fullmatch(r"isic_\d+", stem):
        raise ValueError(f"normalized source identifier is not isic_<digits>: {filename}")
    return stem


def find_cross_split_overlaps(names_by_split, fail=False):
    ids = {split: {normalized_source_id(name) for name in names} for split, names in names_by_split.items()}
    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            common = sorted(ids.get(left, set()) & ids.get(right, set()))
            if common:
                overlaps[f"{left}:{right}"] = common
    if fail and overlaps:
        raise ValueError(f"cross-split overlap: {overlaps}")
    return overlaps


def validate_dataset(root, expected_counts=None, check_images=True):
    root = Path(root)
    counts, classes, names = {}, Counter(), {}
    for split in SPLITS:
        images = {p.stem: p for p in (root / "images" / split).glob("*.jpg")}
        labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
        if set(images) != set(labels):
            raise ValueError(f"{split}: image/label stem mismatch")
        if expected_counts is not None and len(images) != expected_counts[split]:
            raise ValueError(f"{split}: unexpected source count {len(images)} != {expected_counts[split]}")
        if check_images:
            def readable(item):
                stem, image_path = item
                return stem, cv2.imread(str(image_path)) is not None
            with ThreadPoolExecutor(max_workers=8) as executor:
                for stem, is_readable in executor.map(readable, images.items()):
                    if not is_readable:
                        raise ValueError(f"{images[stem]}: JPEG is not readable")
        for stem, image_path in images.items():
            normalized_source_id(stem)
            with labels[stem].open("r", encoding="utf-8") as handle:
                for class_id, _ in parse_yolo_lines(handle, labels[stem]):
                    classes[str(class_id)] += 1
        counts[split] = len(images)
        names[split] = list(images)
    overlaps = find_cross_split_overlaps(names, fail=True)
    return counts, dict(sorted(classes.items())), overlaps


def file_record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def build_manifest(project_root):
    project_root = Path(project_root).resolve()
    provenance_root = (project_root / SOURCE_RELATIVE).resolve()
    manifest = {}
    paths = []
    for dirname in ("data", "runs"):
        base = project_root / dirname
        if base.exists():
            paths.extend(
                p for p in base.rglob("*") if p.is_file()
                and provenance_root not in p.resolve().parents
            )
    paths.sort()
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = executor.map(file_record, paths)
        for path, record in zip(paths, records):
            manifest[path.relative_to(project_root).as_posix()] = record
    return manifest


def compare_manifests(old, new):
    old_keys, new_keys = set(old), set(new)
    common = old_keys & new_keys
    return {
        "added": sorted(new_keys - old_keys),
        "removed": sorted(old_keys - new_keys),
        "size_changed": sorted(k for k in common if old[k]["size"] != new[k]["size"]),
        "digest_changed": sorted(k for k in common if old[k]["size"] == new[k]["size"] and old[k]["sha256"] != new[k]["sha256"]),
    }


def validate_output_path(path, project_root):
    path, project_root = Path(path).resolve(), Path(project_root).resolve()
    audit_root = (project_root / "audit_reports").resolve()
    if path == audit_root or audit_root not in path.parents:
        raise ValueError(f"output must be below {audit_root}")
    return path


def _audit_output(root):
    images, labels, names, classes, eligible, invalid = {}, {}, {}, Counter(), 0, []
    for split in SPLITS:
        image_paths = list((root / "images" / split).glob("*.jpg"))
        label_paths = list((root / "labels" / split).glob("*.txt"))
        if {p.stem for p in image_paths} != {p.stem for p in label_paths}:
            raise ValueError(f"{root.name}/{split}: output image/label stem mismatch")
        images[split], labels[split] = len(image_paths), len(label_paths)
        names[split] = [p.name for p in image_paths]
        for image_path in image_paths:
            normalized_source_id(image_path.name)
            if cv2.imread(str(image_path)) is None:
                invalid.append((image_path, "unreadable JPEG"))
        for label_path in label_paths:
            with label_path.open("r", encoding="utf-8") as handle:
                parsed = parse_yolo_lines(handle, label_path)
                for class_id, _ in parsed:
                    classes[str(class_id)] += 1
                if split == "train" and parsed and all(class_id != 5 for class_id, _ in parsed):
                    eligible += 1
    return images, labels, names, classes, eligible, invalid


def source_split_counts(root):
    root = Path(root)
    counts = {}
    for split in SPLITS:
        images = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        if images != labels:
            raise ValueError(f"{split}: image/label stem mismatch")
        counts[split] = len(images)
    return counts


def audit_project(project_root, expected_counts=EXPECTED_SOURCE_COUNTS):
    project_root = Path(project_root).resolve()
    source_counts = source_split_counts(project_root / SOURCE_RELATIVE)
    if expected_counts is not None and source_counts != expected_counts:
        raise ValueError(f"unexpected source counts: {source_counts} != {expected_counts}")
    classes, eligible_attempt_sources, augmentation_saved, invalid_files = Counter(), 0, 0, []
    missing = [str(project_root / relative) for relative in OUTPUT_RELATIVES if not (project_root / relative).is_dir()]
    if missing:
        raise ValueError(f"missing processed dataset root: {missing}")
    output_images, output_labels, all_names, datasets = {}, {}, {s: [] for s in SPLITS}, {}
    for relative in OUTPUT_RELATIVES:
        root = project_root / relative
        if root.exists():
            images, labels, names, output_classes, eligible, invalid = _audit_output(root)
            output_images[root.name], output_labels[root.name] = images, labels
            composition = {}
            for split in SPLITS:
                composition[split] = {
                    "v1": sum(bool(re.search(r"_v1\.jpg$", n, re.I)) for n in names[split]),
                    "v7": sum(bool(re.search(r"_v7\.jpg$", n, re.I)) for n in names[split]),
                    "aug": sum(bool(re.search(r"_aug\d+\.jpg$", n, re.I)) for n in names[split]),
                }
                composition[split]["other"] = len(names[split]) - sum(composition[split].values())
            datasets[root.name] = {"output_images": images, "output_labels": labels,
                                   "class_polygons": dict(sorted(output_classes.items())),
                                   "view_composition": composition}
            classes.update(output_classes)
            invalid_files.extend(
                {"path": path.relative_to(project_root).as_posix(), "reason": reason}
                for path, reason in invalid
            )
            if root.name == "dataset_yolo_640x640_multiview":
                eligible_attempt_sources = eligible
            if root.name == "dataset_yolo_aug_p2_cbam":
                augmentation_saved = sum(1 for name in names["train"] if re.search(r"_aug\d+\.jpg$", name, re.I))
            for split in SPLITS:
                all_names[split].extend(names[split])
    overlaps = find_cross_split_overlaps(all_names) if any(all_names.values()) else {}
    if overlaps:
        raise ValueError(f"cross-split overlap: {overlaps}")
    observed = output_images["dataset_yolo_aug_p2_cbam"]["train"]
    return {
        "source_pairs": source_counts,
        "output_images": output_images,
        "output_labels": output_labels,
        "invalid_files": sorted(invalid_files, key=lambda item: item["path"]),
        "skipped_files": max(0, eligible_attempt_sources * 3 - augmentation_saved),
        "augmentation_attempts": eligible_attempt_sources * 3,
        "augmentation_saved": augmentation_saved,
        "class_polygons": dict(sorted(classes.items())),
        "cross_split_overlaps": overlaps,
        "datasets": datasets,
        "observed_augmented_train_count": observed,
        "equals_31880": observed == 31880,
    }


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--compare-manifest")
    parser.add_argument("--expected-counts", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    project = Path(args.project_root).resolve()
    report = validate_output_path(args.report, project)
    manifest_path = validate_output_path(args.manifest, project) if args.manifest else None
    compare_path = validate_output_path(args.compare_manifest, project) if args.compare_manifest else None
    expected = EXPECTED_SOURCE_COUNTS
    if args.expected_counts:
        values = [int(value) for value in args.expected_counts.split(",")]
        expected = dict(zip(SPLITS, values, strict=True))
    payload = audit_project(project, expected)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_manifest(project) if manifest_path or compare_path else None
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if compare_path:
        old = json.loads(compare_path.read_text(encoding="utf-8"))
        differences = compare_manifests(old, manifest)
        if any(differences.values()):
            comparison_report = report.with_name(f"{report.stem}.manifest-diff.json")
            comparison_report.write_text(json.dumps(differences, indent=2), encoding="utf-8")
            print(json.dumps(differences, indent=2))
            return 1
        print("Manifest comparison: identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
