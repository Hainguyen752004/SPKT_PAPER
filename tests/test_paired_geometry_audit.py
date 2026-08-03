import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data_processing import audit_paired_geometry as audit_module
from data_processing.audit_paired_geometry import (
    _parse_yolo_label,
    audit_paired_geometry,
    main,
    rasterize_p2_mask,
    write_report_atomic,
)


def _polygon_area(points):
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def _instance(pair_id="case", class_id=2):
    polygon = [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]]
    area = _polygon_area(polygon)
    return {
        "instance_id": f"{pair_id}:0",
        "class_id": class_id,
        "source_polygon": copy.deepcopy(polygon),
        "crop_intersection_polygon": copy.deepcopy(polygon),
        "canvas_polygon": copy.deepcopy(polygon),
        "source_area": area,
        "crop_intersection_area": area,
        "canvas_area": area,
        "status": "kept",
        "reason": None,
    }


def _record(pair_id, view, instance=None):
    instances = [_instance(pair_id)] if instance is None else [instance]
    return {
        "schema_version": 1,
        "pair_id": pair_id,
        "split": "train",
        "view": view,
        "source": f"images/train/{pair_id}.jpg",
        "transform": {
            "schema_version": 1,
            "source_width": 8.0,
            "source_height": 8.0,
            "crop_box": [0.0, 0.0, 8.0, 8.0],
            "scale": 1.0,
            "resized_width": 8,
            "resized_height": 8,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "canvas_width": 8.0,
            "canvas_height": 8.0,
        },
        "preprocessing_version": "geometry_v1",
        "instances": instances,
        "input_instance_count": len(instances),
        "output_instance_count": sum(item["status"] == "kept" for item in instances),
        "dropped_instance_reasons": {},
        "artifact_proxies": {
            "hair_mask_coverage": 0.0,
            "vignette_crop_ratio": 0.0,
            "gray_world_gains": [1.0, 1.0, 1.0],
            "gray_world_correction_magnitude": 0.0,
        },
    }


def _label_line(instance, canvas_width=8.0, canvas_height=8.0):
    coords = []
    for x, y in instance["canvas_polygon"]:
        coords.extend((x / canvas_width, y / canvas_height))
    return f'{instance["class_id"]} ' + " ".join(f"{value:.6f}" for value in coords)


def _write_view(root, record):
    split = record["split"]
    stem = f'{record["pair_id"]}_{record["view"]}'
    image = root / "images" / split / f"{stem}.jpg"
    label = root / "labels" / split / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    canvas_width = int(record["transform"]["canvas_width"])
    canvas_height = int(record["transform"]["canvas_height"])
    assert cv2.imwrite(
        str(image), np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    )
    kept = [item for item in record["instances"] if item["status"] == "kept"]
    label.write_text(
        "\n".join(_label_line(item, canvas_width, canvas_height) for item in kept)
        + ("\n" if kept else ""),
        encoding="utf-8",
    )


def _dataset(tmp_path, records=None):
    root = tmp_path / "dataset"
    records = records or [_record("case", "v1"), _record("case", "v7")]
    for record in records:
        _write_view(root, record)
    metadata = root / "metadata" / "transforms.jsonl"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return root, metadata, records


def test_valid_pair_passes_publication_gate(tmp_path):
    root, metadata, _ = _dataset(tmp_path)

    report = audit_paired_geometry(root, metadata)

    assert report["candidate_pair_count"] == 1
    assert report["avc_valid_pair_count"] == 1
    assert report["avc_excluded_pair_count"] == 0
    assert report["publication_gate_passed"] is True
    assert report["reason_counts"] == {}
    assert report["max_roundtrip_error"] <= 1e-5
    json.dumps(report)


def test_missing_view_is_structural_failure(tmp_path):
    root, metadata, _ = _dataset(tmp_path, [_record("case", "v1")])

    report = audit_paired_geometry(root, metadata)

    assert report["missing_view_count"] == 1
    assert report["reason_counts"]["MISSING_VIEW"] == 1
    assert report["excluded_pair_ids"]["MISSING_VIEW"] == ["case"]
    assert report["publication_gate_passed"] is False


def test_duplicate_record_malformed_json_and_missing_fields_are_reported_without_raising(tmp_path):
    root, metadata, records = _dataset(tmp_path)
    with metadata.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(records[0]) + "\n")
        handle.write("{not-json}\n")
        handle.write(json.dumps({"pair_id": "broken", "split": "train", "view": "v1"}) + "\n")

    report = audit_paired_geometry(root, metadata)

    assert report["duplicate_view_count"] == 1
    assert report["malformed_json_count"] == 1
    assert report["malformed_record_count"] == 1
    assert report["reason_counts"]["DUPLICATE_VIEW"] == 1
    assert report["reason_counts"]["MALFORMED_JSON"] == 1
    assert report["reason_counts"]["MALFORMED_METADATA"] == 1
    assert report["publication_gate_passed"] is False


def test_invalid_utf8_metadata_returns_failure_report_and_cli_writes_it(tmp_path):
    root, metadata, _ = _dataset(tmp_path)
    metadata.write_bytes(b'\xff\xfe{"pair_id":"case"}\n')
    output = tmp_path / "invalid-utf8-report.json"

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["MALFORMED_METADATA"] == 1
    assert report["publication_gate_passed"] is False
    assert main(["--dataset", str(root), "--output", str(output)]) != 0
    assert json.loads(output.read_text(encoding="utf-8"))["reason_counts"]["MALFORMED_METADATA"] == 1


def test_metadata_os_read_error_returns_failure_report(tmp_path, monkeypatch):
    root, metadata, _ = _dataset(tmp_path)
    original_open = audit_module.Path.open

    def fail_metadata_open(path, *args, **kwargs):
        if path == metadata:
            raise OSError("controlled read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(audit_module.Path, "open", fail_metadata_open)

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["MALFORMED_METADATA"] == 1
    assert report["publication_gate_passed"] is False


def test_class_count_and_label_polygon_mismatches_fail_gate(tmp_path):
    root, metadata, records = _dataset(tmp_path)
    records[1]["instances"][0]["class_id"] = 3
    records[1]["input_instance_count"] = 99
    metadata.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    label = root / "labels/train/case_v1.txt"
    label.write_text("2 0.125 0.125 0.75 0.125 0.50 0.75 0.125 0.75\n", encoding="utf-8")

    report = audit_paired_geometry(root, metadata)

    assert report["class_correspondence_error_count"] >= 1
    assert report["instance_correspondence_error_count"] >= 1
    assert report["label_correspondence_error_count"] >= 1
    assert report["publication_gate_passed"] is False


def test_invalid_empty_polygon_is_not_misreported_as_a_p2_pooling_failure(tmp_path):
    records = [_record("case", "v1"), _record("case", "v7")]
    records[1]["instances"][0]["canvas_polygon"] = []
    records[1]["instances"][0]["canvas_area"] = 0.0
    root, metadata, _ = _dataset(tmp_path, records)

    report = audit_paired_geometry(root, metadata)

    assert report["invalid_polygon_count"] >= 1
    assert report["zero_area_polygon_count"] >= 1
    assert report["empty_p2_mask_count"] == 0
    assert report["publication_gate_passed"] is False


def test_source_crop_geometry_mismatch_fails_for_full_v1_crop(tmp_path):
    records = [_record("case", "v1"), _record("case", "v7")]
    fake_crop = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]]
    records[0]["instances"][0]["crop_intersection_polygon"] = copy.deepcopy(fake_crop)
    records[0]["instances"][0]["crop_intersection_area"] = _polygon_area(fake_crop)
    records[0]["instances"][0]["canvas_polygon"] = copy.deepcopy(fake_crop)
    records[0]["instances"][0]["canvas_area"] = _polygon_area(fake_crop)
    root, metadata, _ = _dataset(tmp_path, records)

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["SOURCE_CROP_GEOMETRY_MISMATCH"] >= 1
    assert report["publication_gate_passed"] is False


def test_source_crop_comparison_accepts_cyclic_shift_and_reversed_orientation(tmp_path):
    records = [_record("case", "v1"), _record("case", "v7")]
    polygon = records[0]["instances"][0]["crop_intersection_polygon"]
    reordered = list(reversed(polygon[1:] + polygon[:1]))
    records[0]["instances"][0]["crop_intersection_polygon"] = reordered
    records[0]["instances"][0]["canvas_polygon"] = copy.deepcopy(reordered)
    root, metadata, _ = _dataset(tmp_path, records)

    report = audit_paired_geometry(root, metadata)

    assert "SOURCE_CROP_GEOMETRY_MISMATCH" not in report["reason_counts"]
    assert report["publication_gate_passed"] is True


def test_out_of_canvas_and_finite_zero_area_polygons_are_distinguished(tmp_path):
    outside_records = [_record("outside", "v1"), _record("outside", "v7")]
    outside_records[0]["instances"][0]["canvas_polygon"][0][0] = 9.0
    outside_records[0]["instances"][0]["canvas_area"] = _polygon_area(
        outside_records[0]["instances"][0]["canvas_polygon"]
    )
    outside_root, outside_metadata, _ = _dataset(tmp_path / "outside", outside_records)

    outside_report = audit_paired_geometry(outside_root, outside_metadata)

    assert outside_report["out_of_canvas_polygon_count"] >= 1
    assert outside_report["reason_counts"]["OUT_OF_CANVAS_POLYGON"] >= 1

    zero_records = [_record("zero", "v1"), _record("zero", "v7")]
    line = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    zero_records[0]["instances"][0]["canvas_polygon"] = line
    zero_records[0]["instances"][0]["canvas_area"] = 0.0
    zero_root, zero_metadata, _ = _dataset(tmp_path / "zero", zero_records)

    zero_report = audit_paired_geometry(zero_root, zero_metadata)

    assert zero_report["zero_area_polygon_count"] >= 1
    assert zero_report["reason_counts"]["ZERO_AREA_POLYGON"] >= 1


def test_malformed_yolo_line_is_a_structural_label_failure(tmp_path):
    root, metadata, _ = _dataset(tmp_path)
    (root / "labels/train/case_v1.txt").write_text("2 0.1 0.2\n", encoding="utf-8")

    report = audit_paired_geometry(root, metadata)

    assert report["label_correspondence_error_count"] >= 1
    assert report["reason_counts"]["INVALID_LABEL"] == 1
    assert report["publication_gate_passed"] is False


def test_duplicate_filesystem_views_across_supported_locations_and_extensions(tmp_path):
    root, metadata, _ = _dataset(tmp_path)
    duplicate_image = root / "images/train/case_v1.png"
    assert cv2.imwrite(str(duplicate_image), np.zeros((8, 8, 3), dtype=np.uint8))
    duplicate_label = root / "labels/train_v7_eval/case_v1.txt"
    duplicate_label.parent.mkdir(parents=True)
    duplicate_label.write_text(
        (root / "labels/train/case_v1.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    report = audit_paired_geometry(root, metadata)

    assert report["duplicate_view_count"] == 2
    assert report["reason_counts"]["DUPLICATE_IMAGE_VIEW"] == 1
    assert report["reason_counts"]["DUPLICATE_LABEL_VIEW"] == 1
    assert report["publication_gate_passed"] is False


def test_raster_helper_uses_round_clip_ceil_shape_and_adaptive_max_pooling():
    pooled = rasterize_p2_mask(
        [(-0.1, -0.1), (0.49, 0.49), (1.1, 0.49)],
        canvas_width=5,
        canvas_height=6,
        stride=4,
    )

    assert pooled.dtype == torch.bool
    assert tuple(pooled.shape) == (2, 2)
    assert pooled.tolist() == [[True, True], [True, True]]


def test_tiny_positive_polygon_always_survives_fill_and_max_pooling():
    # fillPoly paints at least the rounded vertex pixel; adaptive max pooling
    # therefore cannot erase any successfully rasterized positive polygon.
    pooled = rasterize_p2_mask(
        [(0.200, 0.200), (0.201, 0.200), (0.200, 0.201)], 8, 8, 4
    )

    assert torch.count_nonzero(pooled).item() == 1


def test_empty_p2_reason_is_wired_to_zero_foreground_for_otherwise_valid_metadata(
        tmp_path, monkeypatch):
    root, metadata, _ = _dataset(tmp_path)
    monkeypatch.setattr(
        audit_module,
        "rasterize_p2_mask",
        lambda *_args, **_kwargs: torch.zeros((2, 2), dtype=torch.bool),
    )

    report = audit_paired_geometry(root, metadata)

    assert report["empty_p2_mask_count"] == 2
    assert report["reason_counts"]["EMPTY_P2_MASK"] == 2
    assert report["publication_gate_passed"] is False


def test_transform_mismatch_and_excessive_roundtrip_are_reported(tmp_path):
    records = [_record("case", "v1"), _record("case", "v7")]
    records[1]["instances"][0]["canvas_polygon"][0][0] = 2.0
    records[1]["instances"][0]["canvas_area"] = _polygon_area(
        records[1]["instances"][0]["canvas_polygon"]
    )
    root, metadata, _ = _dataset(tmp_path, records)

    report = audit_paired_geometry(root, metadata)

    assert report["metadata_transform_mismatch_count"] >= 1
    assert report["max_roundtrip_error"] > 1e-5
    assert report["reason_counts"]["ROUNDTRIP_ERROR"] >= 1
    assert report["publication_gate_passed"] is False


@pytest.mark.parametrize("reason", [
    "INSTANCE_OUTSIDE_V7_CROP",
    "INSTANCE_DEGENERATE_AFTER_CLIP",
])
def test_legitimate_v7_crop_exclusion_does_not_fail_dataset_gate(tmp_path, reason):
    v1 = _record("case", "v1")
    dropped = _instance("case")
    dropped.update({
        "crop_intersection_polygon": [],
        "canvas_polygon": [],
        "crop_intersection_area": 0.0,
        "canvas_area": 0.0,
        "status": "dropped",
        "reason": reason,
    })
    v7 = _record("case", "v7", dropped)
    if reason == "INSTANCE_OUTSIDE_V7_CROP":
        v7["transform"].update({
            "crop_box": [7.0, 7.0, 1.0, 1.0],
            "scale": 8.0,
            "resized_width": 8,
            "resized_height": 8,
        })
    else:
        v7["transform"].update({
            "crop_box": [6.0, 6.0, 2.0, 2.0],
            "scale": 4.0,
            "resized_width": 8,
            "resized_height": 8,
        })
    v7["dropped_instance_reasons"] = {reason: 1}
    root, metadata, _ = _dataset(tmp_path, [v1, v7])

    report = audit_paired_geometry(root, metadata)

    assert report["avc_valid_pair_count"] == 0
    assert report["avc_excluded_pair_count"] == 1
    assert report["excluded_pair_ids"][reason] == ["case"]
    assert report["publication_gate_passed"] is True


def test_false_claim_of_legitimate_crop_exclusion_fails_gate(tmp_path):
    v1 = _record("case", "v1")
    dropped = _instance("case")
    dropped.update({
        "crop_intersection_polygon": [],
        "canvas_polygon": [],
        "crop_intersection_area": 0.0,
        "canvas_area": 0.0,
        "status": "dropped",
        "reason": "INSTANCE_OUTSIDE_V7_CROP",
    })
    v7 = _record("case", "v7", dropped)
    v7["dropped_instance_reasons"] = {"INSTANCE_OUTSIDE_V7_CROP": 1}
    root, metadata, _ = _dataset(tmp_path, [v1, v7])

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["INVALID_DROPPED_INSTANCE"] >= 1
    assert report["publication_gate_passed"] is False


def test_missing_image_and_label_are_counted(tmp_path):
    root, metadata, _ = _dataset(tmp_path)
    (root / "images/train/case_v1.jpg").unlink()
    (root / "labels/train/case_v7.txt").unlink()

    report = audit_paired_geometry(root, metadata)

    assert report["missing_image_count"] == 1
    assert report["missing_label_count"] == 1
    assert report["publication_gate_passed"] is False


def test_atomic_report_write_preserves_old_file_on_serialization_failure(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("old", encoding="utf-8")

    with pytest.raises(TypeError):
        write_report_atomic(output, {"not_serializable": object()})

    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_cli_exit_code_tracks_gate_and_uses_default_metadata(tmp_path):
    root, _, _ = _dataset(tmp_path)
    good_output = tmp_path / "good.json"
    bad_output = tmp_path / "bad.json"

    assert main(["--dataset", str(root), "--output", str(good_output)]) == 0
    assert json.loads(good_output.read_text(encoding="utf-8"))["publication_gate_passed"] is True

    (root / "labels/train/case_v7.txt").unlink()
    assert main(["--dataset", str(root), "--output", str(bad_output), "--p2-stride", "4"]) != 0
    assert json.loads(bad_output.read_text(encoding="utf-8"))["publication_gate_passed"] is False


def test_syntactically_valid_json_with_wrong_field_types_never_crashes(tmp_path):
    mutations = [
        lambda record: record.update(pair_id=[]),
        lambda record: record.update(source=[]),
        lambda record: record.update(preprocessing_version=False),
        lambda record: record.update(transform=[]),
        lambda record: record.update(instances={}),
        lambda record: record.update(dropped_instance_reasons=[]),
        lambda record: record.update(artifact_proxies=[]),
        lambda record: record["instances"][0].update(instance_id=[]),
        lambda record: record["instances"][0].update(status=[]),
        lambda record: record["instances"][0].update(status="dropped", reason=[]),
        lambda record: record["instances"][0].update(class_id=True),
        lambda record: record["instances"][0].update(source_polygon=[[{}, 1], [2, 2], [3, 3]]),
        lambda record: record["transform"].update(canvas_width=True),
        lambda record: record["transform"].update(canvas_width=8.5),
    ]
    for index, mutate in enumerate(mutations):
        root, metadata, records = _dataset(tmp_path / str(index))
        mutate(records[0])
        metadata.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

        report = audit_paired_geometry(root, metadata)

        assert report["reason_counts"]["MALFORMED_METADATA"] >= 1
        assert report["publication_gate_passed"] is False
        json.dumps(report)


@pytest.mark.parametrize("payload", [None, True, 1, "record", [], {}])
def test_every_json_primitive_top_level_returns_a_failure_report(tmp_path, payload):
    root, metadata, _ = _dataset(tmp_path)
    metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["MALFORMED_METADATA"] == 1
    assert report["publication_gate_passed"] is False
    json.dumps(report)


def test_cli_writes_failure_report_for_unhashable_metadata_fields(tmp_path):
    root, metadata, records = _dataset(tmp_path)
    records[0]["instances"][0].update(status="dropped", reason=[])
    metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "malformed-report.json"

    exit_code = main(["--dataset", str(root), "--output", str(output)])

    assert exit_code != 0
    assert json.loads(output.read_text(encoding="utf-8"))["reason_counts"]["MALFORMED_METADATA"] >= 1


def test_six_decimal_label_serialization_collapse_is_rejected(tmp_path):
    tiny = [[1.0, 1.0], [1.0001, 1.0], [1.0, 1.0001]]
    instances = []
    for view in ("v1", "v7"):
        instance = _instance("tiny")
        area = _polygon_area(tiny)
        instance.update({
            "source_polygon": copy.deepcopy(tiny),
            "crop_intersection_polygon": copy.deepcopy(tiny),
            "canvas_polygon": copy.deepcopy(tiny),
            "source_area": area,
            "crop_intersection_area": area,
            "canvas_area": area,
        })
        record = _record("tiny", view, instance)
        record["transform"].update({
            "source_width": 640.0,
            "source_height": 640.0,
            "crop_box": [0.0, 0.0, 640.0, 640.0],
            "resized_width": 640,
            "resized_height": 640,
            "canvas_width": 640.0,
            "canvas_height": 640.0,
        })
        instances.append(record)
    root, metadata, _ = _dataset(tmp_path, instances)

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["ZERO_AREA_LABEL_POLYGON"] == 2
    assert report["zero_area_polygon_count"] == 2
    assert report["publication_gate_passed"] is False


def test_every_zero_area_label_polygon_in_one_file_is_counted(tmp_path):
    root, metadata, _ = _dataset(tmp_path)
    (root / "labels/train/case_v1.txt").write_text(
        "2 0.1 0.1 0.2 0.2 0.3 0.3\n"
        "3 0.4 0.4 0.4 0.4 0.4 0.4\n",
        encoding="utf-8",
    )

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["ZERO_AREA_LABEL_POLYGON"] == 2
    assert report["zero_area_polygon_count"] == 2
    assert report["publication_gate_passed"] is False


def test_tiny_positive_serialized_label_polygon_is_accepted_and_raster_survives(tmp_path):
    label = tmp_path / "tiny.txt"
    label.write_text(
        "2 0.100000 0.100000 0.100001 0.100000 0.100000 0.100001\n",
        encoding="utf-8",
    )

    parsed = _parse_yolo_label(label)
    pooled = rasterize_p2_mask(parsed[0][1], 8, 8, 4)

    assert len(parsed) == 1
    assert _polygon_area(parsed[0][1]) > 0.0
    assert torch.count_nonzero(pooled).item() == 1


def test_extreme_finite_transform_values_return_malformed_report_without_overflow(tmp_path):
    root, metadata, records = _dataset(tmp_path)
    records[0]["transform"].update({
        "source_width": 1e308,
        "source_height": 1e308,
        "crop_box": [0.0, 0.0, 1e308, 1e308],
        "scale": 1e308,
    })
    metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["MALFORMED_METADATA"] >= 1
    assert report["publication_gate_passed"] is False


def test_pair_views_must_match_source_path_dimensions_and_preprocessing_version(tmp_path):
    records = [_record("identity", "v1"), _record("identity", "v7")]
    records[1]["source"] = "images/train/different.jpg"
    records[1]["preprocessing_version"] = "geometry_v2"
    records[1]["transform"].update({
        "source_width": 9.0,
        "source_height": 9.0,
        "crop_box": [0.0, 0.0, 9.0, 9.0],
        "scale": 8 / 9,
        "resized_width": 8,
        "resized_height": 8,
    })
    root, metadata, _ = _dataset(tmp_path, records)

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["SOURCE_IDENTITY_MISMATCH"] == 1
    assert report["reason_counts"]["SOURCE_DIMENSION_MISMATCH"] == 1
    assert report["reason_counts"]["PREPROCESSING_VERSION_MISMATCH"] == 1
    assert report["publication_gate_passed"] is False


def test_v1_requires_full_source_crop_and_neutral_artifact_proxies(tmp_path):
    cropped_records = [_record("crop", "v1"), _record("crop", "v7")]
    cropped_records[0]["transform"].update({
        "crop_box": [1.0, 1.0, 7.0, 7.0],
        "scale": 8 / 7,
        "resized_width": 8,
        "resized_height": 8,
    })
    cropped_root, cropped_metadata, _ = _dataset(tmp_path / "crop", cropped_records)
    proxy_records = [_record("proxy", "v1"), _record("proxy", "v7")]
    proxy_records[0]["artifact_proxies"]["hair_mask_coverage"] = 0.25
    proxy_root, proxy_metadata, _ = _dataset(tmp_path / "proxy", proxy_records)

    cropped_report = audit_paired_geometry(cropped_root, cropped_metadata)
    proxy_report = audit_paired_geometry(proxy_root, proxy_metadata)

    assert cropped_report["reason_counts"]["V1_TRANSFORM_SEMANTICS"] >= 1
    assert proxy_report["reason_counts"]["V1_TRANSFORM_SEMANTICS"] >= 1
    assert cropped_report["publication_gate_passed"] is False
    assert proxy_report["publication_gate_passed"] is False


def test_pair_id_view_duplicates_are_global_but_val_errors_do_not_exclude_train_pair(tmp_path):
    records = [_record("same", "v1"), _record("same", "v7")]
    val = _record("same", "v1")
    val["split"] = "val"
    records.append(val)
    root, metadata, _ = _dataset(tmp_path, records)
    (root / "labels/val/same_v1.txt").write_text("not a yolo polygon\n", encoding="utf-8")

    report = audit_paired_geometry(root, metadata)

    assert report["duplicate_view_count"] == 1
    assert report["reason_counts"]["DUPLICATE_VIEW"] == 1
    assert report["excluded_pair_ids"]["INVALID_LABEL"] == ["val/same"]
    assert report["avc_valid_pair_count"] == 1
    assert report["avc_excluded_pair_count"] == 0
    assert report["publication_gate_passed"] is False


def test_huge_canvas_is_rejected_before_any_raster_allocation(tmp_path, monkeypatch):
    records = [_record("huge", "v1"), _record("huge", "v7")]
    root, metadata, _ = _dataset(tmp_path, records)
    for record in records:
        record["transform"].update({"canvas_width": 9000, "canvas_height": 9000})
    metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    def forbidden_allocation(*_args, **_kwargs):
        raise AssertionError("raster allocation must not run")

    monkeypatch.setattr(audit_module.np, "zeros", forbidden_allocation)

    report = audit_paired_geometry(root, metadata)

    assert report["reason_counts"]["CANVAS_DIMENSION_LIMIT"] == 2
    assert report["publication_gate_passed"] is False


def test_corrupt_and_dimension_mismatched_images_fail_structurally(tmp_path):
    corrupt_root, corrupt_metadata, _ = _dataset(tmp_path / "corrupt")
    (corrupt_root / "images/train/case_v1.jpg").write_bytes(b"not-an-image")
    mismatch_root, mismatch_metadata, _ = _dataset(tmp_path / "mismatch")
    mismatch_image = mismatch_root / "images/train/case_v1.jpg"
    assert cv2.imwrite(str(mismatch_image), np.zeros((7, 8, 3), dtype=np.uint8))

    corrupt_report = audit_paired_geometry(corrupt_root, corrupt_metadata)
    mismatch_report = audit_paired_geometry(mismatch_root, mismatch_metadata)

    assert corrupt_report["reason_counts"]["CORRUPT_IMAGE"] == 1
    assert mismatch_report["reason_counts"]["IMAGE_DIMENSION_MISMATCH"] == 1
    assert corrupt_report["publication_gate_passed"] is False
    assert mismatch_report["publication_gate_passed"] is False


def test_atomic_report_cleanup_when_temp_write_or_replace_fails(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("old", encoding="utf-8")
    original_fdopen = audit_module.os.fdopen

    def fail_fdopen(*_args, **_kwargs):
        raise OSError("controlled temp write failure")

    monkeypatch.setattr(audit_module.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="temp write"):
        write_report_atomic(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []

    monkeypatch.setattr(audit_module.os, "fdopen", original_fdopen)
    monkeypatch.setattr(
        audit_module.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("controlled replace failure")),
    )
    with pytest.raises(OSError, match="replace"):
        write_report_atomic(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_cli_returns_nonzero_if_atomic_replace_fails(tmp_path, monkeypatch):
    root, _, _ = _dataset(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        audit_module.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("controlled replace failure")),
    )

    assert main(["--dataset", str(root), "--output", str(output)]) != 0
    assert output.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".report.json.*.tmp")) == []
