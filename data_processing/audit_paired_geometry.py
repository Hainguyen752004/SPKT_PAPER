"""Read-only geometry and correspondence audit for paired v1/v7 datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

try:
    from .paired_geometry import (
        ViewTransform,
        clip_polygon_to_rect,
        polygon_area,
        polygon_intersects_rect,
    )
except ImportError:  # Support direct execution as a script.
    from paired_geometry import (
        ViewTransform,
        clip_polygon_to_rect,
        polygon_area,
        polygon_intersects_rect,
    )


VIEWS = ("v1", "v7")
SPLITS = ("train", "val", "test")
ROUNDTRIP_TOLERANCE = 1e-5
GEOMETRY_TOLERANCE = 1e-5
LABEL_TOLERANCE = 1.1e-6
MAX_CANVAS_DIMENSION = 8192
MAX_CANVAS_PIXELS = 64 * 1024 * 1024
LEGITIMATE_CROP_REASONS = {
    "INSTANCE_OUTSIDE_V7_CROP",
    "INSTANCE_DEGENERATE_AFTER_CLIP",
}
_VIEW_STEM = re.compile(r"^(.+)_(v1|v7)$", re.IGNORECASE)


def _new_report():
    return {
        "candidate_pair_count": 0,
        "avc_valid_pair_count": 0,
        "avc_excluded_pair_count": 0,
        "missing_view_count": 0,
        "duplicate_view_count": 0,
        "missing_image_count": 0,
        "missing_label_count": 0,
        "malformed_json_count": 0,
        "malformed_record_count": 0,
        "instance_correspondence_error_count": 0,
        "class_correspondence_error_count": 0,
        "label_correspondence_error_count": 0,
        "invalid_polygon_count": 0,
        "zero_area_polygon_count": 0,
        "out_of_canvas_polygon_count": 0,
        "empty_p2_mask_count": 0,
        "metadata_transform_mismatch_count": 0,
        "roundtrip_error_count": 0,
        "max_roundtrip_error": 0.0,
        "mean_roundtrip_error": 0.0,
        "reason_counts": {},
        "excluded_pair_ids": {},
        "publication_gate_passed": False,
    }


class _AuditState:
    def __init__(self):
        self.report = _new_report()
        self.reasons = Counter()
        self.excluded = defaultdict(set)
        self.structural_failure = False
        self.roundtrip_errors = []

    def issue(self, reason, pair_key=None, *, structural=True, amount=1):
        self.reasons[reason] += amount
        if pair_key is not None:
            self.excluded[reason].add(pair_key)
        if structural:
            self.structural_failure = True

    def increment(self, field, reason, pair_key=None, *, structural=True, amount=1):
        self.report[field] += amount
        self.issue(reason, pair_key, structural=structural, amount=amount)

    def finish(self):
        if self.roundtrip_errors:
            self.report["max_roundtrip_error"] = float(max(self.roundtrip_errors))
            self.report["mean_roundtrip_error"] = float(
                sum(self.roundtrip_errors) / len(self.roundtrip_errors)
            )
        self.report["reason_counts"] = dict(sorted(self.reasons.items()))
        self.report["excluded_pair_ids"] = {
            reason: sorted(
                pair_id if split == "train" else f"{split}/{pair_id}"
                for split, pair_id in pair_keys
            )
            for reason, pair_keys in sorted(self.excluded.items())
        }
        self.report["publication_gate_passed"] = not self.structural_failure
        return self.report


def _record_identity(value):
    if not isinstance(value, dict):
        return None, None, None
    pair_id = value.get("pair_id")
    split = value.get("split")
    view = value.get("view")
    if not isinstance(pair_id, str) or not pair_id:
        pair_id = None
    if not isinstance(split, str) or split not in SPLITS:
        split = None
    if not isinstance(view, str) or view not in VIEWS:
        view = None
    return pair_id, split, view


def _load_metadata(path, state):
    records = {}
    global_views = set()
    path = Path(path)
    if not path.is_file():
        state.increment("malformed_record_count", "MISSING_METADATA_FILE")
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    state.increment("malformed_json_count", "MALFORMED_JSON")
                    continue
                pair_id, split, view = _record_identity(value)
                if pair_id is None or split is None or view is None:
                    pair_key = (split, pair_id) if split is not None and pair_id is not None else None
                    state.increment("malformed_record_count", "MALFORMED_METADATA", pair_key)
                    continue
                key = (split, pair_id, view)
                pair_key = (split, pair_id)
                global_key = (pair_id, view)
                if global_key in global_views:
                    state.increment("duplicate_view_count", "DUPLICATE_VIEW", pair_key)
                else:
                    global_views.add(global_key)
                if key in records:
                    continue
                records[key] = value
    except (OSError, UnicodeDecodeError):
        # A partial prefix is not trustworthy after a stream-level read failure.
        records.clear()
        state.increment("malformed_record_count", "MALFORMED_METADATA")
    return records


def _scan_view_files(base, suffixes):
    found = defaultdict(list)
    base = Path(base)
    if not base.is_dir():
        return found
    suffixes = {suffix.casefold() for suffix in suffixes}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue
        match = _VIEW_STEM.fullmatch(path.stem)
        if not match:
            continue
        directory = path.parent.name.casefold()
        split = directory.removesuffix("_v7_eval")
        if split in SPLITS:
            found[(split, match.group(1), match.group(2).casefold())].append(path)
    return found


def _finite_polygon(value):
    if not isinstance(value, list) or len(value) < 3:
        return None
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        try:
            vertex = (float(x), float(y))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(coordinate) for coordinate in vertex):
            return None
        points.append(vertex)
    return points


def _points_match(first, second, tolerance=GEOMETRY_TOLERANCE):
    if len(first) != len(second):
        return False
    return all(
        math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance
        for a, b in zip(first, second)
    )


def _polygons_equivalent(first, second, tolerance=GEOMETRY_TOLERANCE):
    """Compare polygon rings independent of starting vertex and orientation."""
    if len(first) != len(second) or not first:
        return False
    if abs(polygon_area(first) - polygon_area(second)) > tolerance:
        return False
    for candidate in (second, list(reversed(second))):
        for offset in range(len(candidate)):
            shifted = candidate[offset:] + candidate[:offset]
            if _points_match(first, shifted, tolerance):
                return True
    return False


def _valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _polygon_payload_is_numeric(value):
    return (
        isinstance(value, list)
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(_valid_number(coordinate) for coordinate in point)
            for point in value
        )
    )


def _metadata_record_is_well_typed(record):
    required = {
        "schema_version", "pair_id", "split", "view", "source", "transform",
        "preprocessing_version", "instances", "input_instance_count",
        "output_instance_count", "dropped_instance_reasons", "artifact_proxies",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        return False
    if (
        isinstance(record["schema_version"], bool)
        or record["schema_version"] != 1
        or not isinstance(record["pair_id"], str)
        or not record["pair_id"]
        or record["split"] not in SPLITS
        or record["view"] not in VIEWS
        or not isinstance(record["source"], str)
        or not record["source"]
        or not isinstance(record["preprocessing_version"], str)
        or not record["preprocessing_version"]
        or not isinstance(record["instances"], list)
        or not _valid_nonnegative_integer(record["input_instance_count"])
        or not _valid_nonnegative_integer(record["output_instance_count"])
        or not isinstance(record["dropped_instance_reasons"], dict)
        or not isinstance(record["artifact_proxies"], dict)
    ):
        return False
    if not all(
        isinstance(reason, str)
        and reason
        and _valid_nonnegative_integer(count)
        for reason, count in record["dropped_instance_reasons"].items()
    ):
        return False

    transform = record["transform"]
    transform_fields = {
        "schema_version", "source_width", "source_height", "crop_box", "scale",
        "resized_width", "resized_height", "pad_x", "pad_y", "canvas_width",
        "canvas_height",
    }
    if not isinstance(transform, dict) or not transform_fields.issubset(transform):
        return False
    if (
        isinstance(transform["schema_version"], bool)
        or transform["schema_version"] != 1
        or not all(_valid_number(transform[field]) for field in (
            "source_width", "source_height", "scale", "pad_x", "pad_y",
            "canvas_width", "canvas_height",
        ))
        or not all(
            isinstance(transform[field], int)
            and not isinstance(transform[field], bool)
            for field in ("resized_width", "resized_height")
        )
        or not isinstance(transform["crop_box"], list)
        or len(transform["crop_box"]) != 4
        or not all(_valid_number(value) for value in transform["crop_box"])
    ):
        return False
    crop_x, crop_y, crop_width, crop_height = map(float, transform["crop_box"])
    scale = float(transform["scale"])
    derived_values = (
        crop_x + crop_width,
        crop_y + crop_height,
        crop_width * scale,
        crop_height * scale,
    )
    if not all(math.isfinite(value) for value in derived_values):
        return False
    for field in ("canvas_width", "canvas_height"):
        value = float(transform[field])
        if value <= 0.0 or not value.is_integer():
            return False

    proxy = record["artifact_proxies"]
    proxy_fields = {
        "hair_mask_coverage", "vignette_crop_ratio", "gray_world_gains",
        "gray_world_correction_magnitude",
    }
    if not proxy_fields.issubset(proxy):
        return False
    if (
        not all(_valid_number(proxy[field]) for field in (
            "hair_mask_coverage", "vignette_crop_ratio",
            "gray_world_correction_magnitude",
        ))
        or not isinstance(proxy["gray_world_gains"], list)
        or len(proxy["gray_world_gains"]) != 3
        or not all(_valid_number(value) for value in proxy["gray_world_gains"])
    ):
        return False

    instance_fields = {
        "instance_id", "class_id", "source_polygon", "crop_intersection_polygon",
        "canvas_polygon", "source_area", "crop_intersection_area", "canvas_area",
        "status", "reason",
    }
    for instance in record["instances"]:
        if not isinstance(instance, dict) or not instance_fields.issubset(instance):
            return False
        status = instance["status"]
        reason = instance["reason"]
        if (
            not isinstance(instance["instance_id"], str)
            or not instance["instance_id"]
            or not isinstance(instance["class_id"], int)
            or isinstance(instance["class_id"], bool)
            or not isinstance(status, str)
            or status not in {"kept", "dropped"}
            or (status == "kept" and reason is not None)
            or (status == "dropped" and (not isinstance(reason, str) or not reason))
            or not all(_polygon_payload_is_numeric(instance[field]) for field in (
                "source_polygon", "crop_intersection_polygon", "canvas_polygon",
            ))
            or not all(_valid_number(instance[field]) for field in (
                "source_area", "crop_intersection_area", "canvas_area",
            ))
        ):
            return False
    return True


def _canvas_dimensions(transform):
    width = int(transform.canvas_width)
    height = int(transform.canvas_height)
    if (
        width > MAX_CANVAS_DIMENSION
        or height > MAX_CANVAS_DIMENSION
        or width * height > MAX_CANVAS_PIXELS
    ):
        return None
    return width, height


def _v1_semantics_are_valid(record, transform):
    if transform.crop_box != (
        0.0, 0.0, transform.source_width, transform.source_height
    ):
        return False
    proxy = record["artifact_proxies"]
    return (
        proxy["hair_mask_coverage"] == 0.0
        and proxy["vignette_crop_ratio"] == 0.0
        and proxy["gray_world_gains"] == [1.0, 1.0, 1.0]
        and proxy["gray_world_correction_magnitude"] == 0.0
    )


def _validate_polygon(instance, field, area_field, bounds, pair_id, state, *, allow_empty=False):
    raw = instance.get(field)
    if allow_empty and raw == []:
        area_value = instance.get(area_field)
        if not _valid_number(area_value) or abs(float(area_value)) > GEOMETRY_TOLERANCE:
            state.increment(
                "metadata_transform_mismatch_count", "METADATA_POLYGON_AREA_MISMATCH", pair_id
            )
        return []
    points = _finite_polygon(raw)
    if points is None:
        state.increment("invalid_polygon_count", "INVALID_POLYGON", pair_id)
        state.increment("zero_area_polygon_count", "ZERO_AREA_POLYGON", pair_id)
        return None
    area = polygon_area(points)
    if area <= 0.0:
        state.increment("zero_area_polygon_count", "ZERO_AREA_POLYGON", pair_id)
    recorded_area = instance.get(area_field)
    if not _valid_number(recorded_area) or abs(float(recorded_area) - area) > GEOMETRY_TOLERANCE:
        state.increment(
            "metadata_transform_mismatch_count", "METADATA_POLYGON_AREA_MISMATCH", pair_id
        )
    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
        if any(not (x_min <= x <= x_max and y_min <= y <= y_max) for x, y in points):
            state.increment("out_of_canvas_polygon_count", "OUT_OF_CANVAS_POLYGON", pair_id)
    return points


class _LabelError(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class _LabelErrors(ValueError):
    def __init__(self, reason_counts):
        super().__init__("label file contains invalid polygons")
        self.reason_counts = Counter(reason_counts)


def _parse_yolo_label(path):
    polygons = []
    errors = Counter()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            parts = raw.split()
            if not parts:
                continue
            try:
                if len(parts) < 7 or (len(parts) - 1) % 2:
                    raise _LabelError("INVALID_LABEL", "invalid polygon coordinate count")
                if not re.fullmatch(r"[+-]?\d+", parts[0]):
                    raise _LabelError("INVALID_LABEL", "class is not an integer")
                class_id = int(parts[0])
                if not 0 <= class_id <= 6:
                    raise _LabelError("INVALID_LABEL", "class outside 0..6")
                try:
                    coordinates = [float(item) for item in parts[1:]]
                except ValueError as error:
                    raise _LabelError("INVALID_LABEL", "non-numeric coordinate") from error
                if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in coordinates):
                    raise _LabelError("INVALID_LABEL", "coordinate outside normalized canvas")
                polygon = list(zip(coordinates[::2], coordinates[1::2]))
                if len(set(polygon)) < 3 or polygon_area(polygon) <= 0.0:
                    raise _LabelError(
                        "ZERO_AREA_LABEL_POLYGON",
                        "polygon collapses after label serialization",
                    )
                polygons.append((class_id, polygon))
            except _LabelError as error:
                errors[error.reason] += 1
    if errors:
        raise _LabelErrors(errors)
    return polygons


def rasterize_p2_mask(normalized_polygon, canvas_width, canvas_height, stride=4):
    """Rasterize a normalized polygon and return its boolean P2 pooled mask."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    width = int(canvas_width)
    height = int(canvas_height)
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")
    vertices = np.asarray([
        [
            np.clip(np.rint(x * width), 0, width - 1),
            np.clip(np.rint(y * height), 0, height - 1),
        ]
        for x, y in normalized_polygon
    ], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(vertices) >= 3:
        cv2.fillPoly(mask, [vertices], 1)
    target = (math.ceil(height / stride), math.ceil(width / stride))
    tensor = torch.from_numpy(mask).to(dtype=torch.float32)[None, None]
    pooled = torch_functional.adaptive_max_pool2d(tensor, target)
    return (pooled > 0)[0, 0]


def _p2_mask_is_empty(normalized_polygon, canvas_width, canvas_height, stride):
    pooled = rasterize_p2_mask(
        normalized_polygon, canvas_width, canvas_height, stride
    )
    return int(torch.count_nonzero(pooled).item()) == 0


def _validate_record(record, label_path, image_path, record_key, p2_stride, state):
    split, pair_id, _view = record_key
    pair_key = (split, pair_id)
    if not _metadata_record_is_well_typed(record):
        state.increment("malformed_record_count", "MALFORMED_METADATA", pair_key)
        return None
    try:
        transform = ViewTransform.from_dict(record)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        state.increment("malformed_record_count", "MALFORMED_METADATA", pair_key)
        return None
    canvas_dimensions = _canvas_dimensions(transform)
    if canvas_dimensions is None:
        state.issue("CANVAS_DIMENSION_LIMIT", pair_key)
        return None
    canvas_width, canvas_height = canvas_dimensions
    if record["view"] == "v1" and not _v1_semantics_are_valid(record, transform):
        state.issue("V1_TRANSFORM_SEMANTICS", pair_key)

    if image_path is not None:
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error):
            image = None
        if image is None:
            state.issue("CORRUPT_IMAGE", pair_key)
        elif image.shape[:2] != (canvas_height, canvas_width):
            state.issue("IMAGE_DIMENSION_MISMATCH", pair_key)
    instances = record.get("instances")

    kept = []
    dropped_reasons = Counter()
    seen_instance_ids = set()
    source_bounds = (0.0, 0.0, transform.source_width, transform.source_height)
    crop_x, crop_y, crop_width, crop_height = transform.crop_box
    crop_bounds = (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
    canvas_bounds = (0.0, 0.0, transform.canvas_width, transform.canvas_height)
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            state.increment("malformed_record_count", "MALFORMED_METADATA", pair_key)
            continue
        instance_id = instance.get("instance_id")
        if instance_id != f"{pair_id}:{index}" or instance_id in seen_instance_ids:
            state.increment(
                "instance_correspondence_error_count", "INSTANCE_MISMATCH", pair_key
            )
        if isinstance(instance_id, str):
            seen_instance_ids.add(instance_id)
        class_id = instance.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id <= 6:
            state.increment("class_correspondence_error_count", "CLASS_MISMATCH", pair_key)

        status = instance.get("status")
        reason = instance.get("reason")
        source = _validate_polygon(
            instance, "source_polygon", "source_area", source_bounds, pair_key, state
        )
        if status == "dropped":
            dropped_reasons[reason] += 1
            crop = _validate_polygon(
                instance, "crop_intersection_polygon", "crop_intersection_area", crop_bounds,
                pair_key, state, allow_empty=True,
            )
            canvas = _validate_polygon(
                instance, "canvas_polygon", "canvas_area", canvas_bounds,
                pair_key, state, allow_empty=True,
            )
            outside_crop = (
                source is not None
                and not polygon_intersects_rect(source, *crop_bounds)
            )
            degenerate_after_clip = (
                source is not None
                and not outside_crop
                and clip_polygon_to_rect(source, *crop_bounds) == []
            )
            reason_matches_geometry = (
                reason == "INSTANCE_OUTSIDE_V7_CROP" and outside_crop
            ) or (
                reason == "INSTANCE_DEGENERATE_AFTER_CLIP" and degenerate_after_clip
            )
            legitimate = (
                record["view"] == "v7"
                and reason in LEGITIMATE_CROP_REASONS
                and reason_matches_geometry
            )
            if not legitimate or crop != [] or canvas != []:
                state.increment(
                    "instance_correspondence_error_count", "INVALID_DROPPED_INSTANCE", pair_key
                )
            else:
                state.issue(reason, pair_key, structural=False)
            continue
        if status != "kept" or reason is not None:
            state.increment("instance_correspondence_error_count", "INSTANCE_MISMATCH", pair_key)
            continue

        crop = _validate_polygon(
            instance, "crop_intersection_polygon", "crop_intersection_area", crop_bounds,
            pair_key, state,
        )
        canvas = _validate_polygon(
            instance, "canvas_polygon", "canvas_area", canvas_bounds, pair_key, state
        )
        kept.append(instance)
        if crop is None or canvas is None:
            continue
        if source is not None:
            expected_crop = clip_polygon_to_rect(source, *crop_bounds)
            if not _polygons_equivalent(expected_crop, crop):
                state.increment(
                    "metadata_transform_mismatch_count",
                    "SOURCE_CROP_GEOMETRY_MISMATCH",
                    pair_key,
                )
        expected_canvas = transform.source_to_canvas(crop)
        if not _points_match(canvas, expected_canvas):
            state.increment(
                "metadata_transform_mismatch_count", "METADATA_TRANSFORM_MISMATCH", pair_key
            )
        recovered = transform.canvas_to_source(canvas)
        if len(recovered) == len(crop):
            errors = [
                math.hypot(actual[0] - expected[0], actual[1] - expected[1])
                for actual, expected in zip(recovered, crop)
            ]
            state.roundtrip_errors.extend(errors)
            if errors and max(errors) > ROUNDTRIP_TOLERANCE:
                state.increment("roundtrip_error_count", "ROUNDTRIP_ERROR", pair_key)
        else:
            state.increment("roundtrip_error_count", "ROUNDTRIP_ERROR", pair_key)

    if record.get("input_instance_count") != len(instances):
        state.increment("instance_correspondence_error_count", "INSTANCE_COUNT_MISMATCH", pair_key)
    if record.get("output_instance_count") != len(kept):
        state.increment("instance_correspondence_error_count", "INSTANCE_COUNT_MISMATCH", pair_key)
    serialized_drops = record.get("dropped_instance_reasons")
    if not isinstance(serialized_drops, dict) or dict(sorted(dropped_reasons.items())) != serialized_drops:
        state.increment("instance_correspondence_error_count", "DROPPED_REASON_MISMATCH", pair_key)

    labels = None
    if label_path is not None:
        try:
            labels = _parse_yolo_label(label_path)
        except _LabelErrors as error:
            for reason, count in sorted(error.reason_counts.items()):
                if reason == "ZERO_AREA_LABEL_POLYGON":
                    state.report["zero_area_polygon_count"] += count
                state.increment(
                    "label_correspondence_error_count", reason, pair_key, amount=count
                )
        except (OSError, UnicodeDecodeError):
            state.increment("label_correspondence_error_count", "INVALID_LABEL", pair_key)
    if labels is not None:
        if len(labels) != len(kept):
            state.increment(
                "label_correspondence_error_count", "LABEL_METADATA_MISMATCH", pair_key
            )
        for (label_class, normalized), instance in zip(labels, kept):
            if label_class != instance.get("class_id"):
                state.increment("class_correspondence_error_count", "CLASS_MISMATCH", pair_key)
                state.increment(
                    "label_correspondence_error_count", "LABEL_METADATA_MISMATCH", pair_key
                )
            canvas = _finite_polygon(instance.get("canvas_polygon"))
            expected = [] if canvas is None else [
                (x / transform.canvas_width, y / transform.canvas_height) for x, y in canvas
            ]
            if not _points_match(normalized, expected, LABEL_TOLERANCE):
                state.increment(
                    "label_correspondence_error_count", "LABEL_METADATA_MISMATCH", pair_key
                )
            if _p2_mask_is_empty(normalized, transform.canvas_width, transform.canvas_height, p2_stride):
                state.increment("empty_p2_mask_count", "EMPTY_P2_MASK", pair_key)
    return {
        "transform": transform,
        "source": record["source"],
        "preprocessing_version": record["preprocessing_version"],
        "instances": instances,
        "kept": kept,
        "dropped_reasons": dropped_reasons,
    }


def _compare_pair_views(pair_key, first, second, state):
    if first["source"] != second["source"]:
        state.issue("SOURCE_IDENTITY_MISMATCH", pair_key)
    if (
        first["transform"].source_width != second["transform"].source_width
        or first["transform"].source_height != second["transform"].source_height
    ):
        state.issue("SOURCE_DIMENSION_MISMATCH", pair_key)
    if first["preprocessing_version"] != second["preprocessing_version"]:
        state.issue("PREPROCESSING_VERSION_MISMATCH", pair_key)
    first_instances = first["instances"]
    second_instances = second["instances"]
    first_by_id = {
        item.get("instance_id"): item for item in first_instances if isinstance(item, dict)
    }
    second_by_id = {
        item.get("instance_id"): item for item in second_instances if isinstance(item, dict)
    }
    if set(first_by_id) != set(second_by_id) or len(first_instances) != len(second_instances):
        state.increment("instance_correspondence_error_count", "INSTANCE_MISMATCH", pair_key)
    for instance_id in sorted(set(first_by_id) & set(second_by_id), key=str):
        left, right = first_by_id[instance_id], second_by_id[instance_id]
        if left.get("class_id") != right.get("class_id"):
            state.increment("class_correspondence_error_count", "CLASS_MISMATCH", pair_key)
        left_source = _finite_polygon(left.get("source_polygon"))
        right_source = _finite_polygon(right.get("source_polygon"))
        if left_source is None or right_source is None or not _points_match(left_source, right_source):
            state.increment(
                "instance_correspondence_error_count", "SOURCE_POLYGON_MISMATCH", pair_key
            )


def audit_paired_geometry(dataset_root, metadata_path, p2_stride=4):
    """Audit paired geometry without modifying any dataset artifact."""
    if isinstance(p2_stride, bool) or not isinstance(p2_stride, int) or p2_stride <= 0:
        raise ValueError("p2_stride must be a positive integer")
    dataset_root = Path(dataset_root)
    state = _AuditState()
    records = _load_metadata(metadata_path, state)
    images = _scan_view_files(
        dataset_root / "images",
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    )
    labels = _scan_view_files(dataset_root / "labels", (".txt",))

    for key, paths in images.items():
        if len(paths) > 1:
            state.increment(
                "duplicate_view_count", "DUPLICATE_IMAGE_VIEW", key[:2], amount=len(paths) - 1
            )
    for key, paths in labels.items():
        if len(paths) > 1:
            state.increment(
                "duplicate_view_count", "DUPLICATE_LABEL_VIEW", key[:2], amount=len(paths) - 1
            )

    all_keys = set(records) | set(images) | set(labels)
    for split, pair_id, view in sorted(all_keys):
        key = (split, pair_id, view)
        pair_key = (split, pair_id)
        if key not in records:
            state.issue("MISSING_METADATA_RECORD", pair_key)
        if key not in images:
            state.increment("missing_image_count", "MISSING_IMAGE", pair_key)
        if key not in labels:
            state.increment("missing_label_count", "MISSING_LABEL", pair_key)

    train_pair_keys = {
        (split, pair_id) for split, pair_id, _view in all_keys if split == "train"
    }
    state.report["candidate_pair_count"] = len(train_pair_keys)
    for pair_key in sorted(train_pair_keys):
        split, pair_id = pair_key
        for view in VIEWS:
            if (split, pair_id, view) not in records:
                state.increment("missing_view_count", "MISSING_VIEW", pair_key)

    validated = {}
    for key, record in sorted(records.items()):
        label_paths = labels.get(key, [])
        image_paths = images.get(key, [])
        validated[key] = _validate_record(
            record, label_paths[0] if len(label_paths) == 1 else None,
            image_paths[0] if len(image_paths) == 1 else None,
            key, p2_stride, state,
        )

    for pair_key in sorted(train_pair_keys):
        split, pair_id = pair_key
        first = validated.get((split, pair_id, "v1"))
        second = validated.get((split, pair_id, "v7"))
        if first is not None and second is not None:
            _compare_pair_views(pair_key, first, second, state)

    excluded_pairs = set().union(*state.excluded.values()) if state.excluded else set()
    state.report["avc_valid_pair_count"] = len(train_pair_keys - excluded_pairs)
    state.report["avc_excluded_pair_count"] = len(excluded_pairs & train_pair_keys)
    return state.finish()


def write_report_atomic(output_path, report):
    """Serialize then atomically replace a report, leaving no partial output."""
    output_path = Path(output_path)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--p2-stride", type=int, default=4)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    dataset = Path(args.dataset)
    metadata = Path(args.metadata) if args.metadata else dataset / "metadata" / "transforms.jsonl"
    report = audit_paired_geometry(dataset, metadata, p2_stride=args.p2_stride)
    try:
        write_report_atomic(args.output, report)
    except OSError:
        return 2
    return 0 if report["publication_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
