import json
import math

import pytest

from data_processing.paired_geometry import ViewTransform


def test_round_trip_preserves_non_clipped_points_within_tolerance():
    transform = ViewTransform(
        source_width=4000,
        source_height=3000,
        crop_box=(250, 125, 2000, 1500),
        scale=0.25,
        resized_width=500,
        resized_height=375,
        pad_x=12.5,
        pad_y=8.75,
        canvas_width=640,
        canvas_height=480,
    )
    source_points = [(250.125, 125.875), (1999.5, 1500.25), (2249.999, 1624.001)]

    canvas_points = transform.source_to_canvas(source_points)
    recovered_points = transform.canvas_to_source(canvas_points)

    assert recovered_points == pytest.approx(source_points, abs=1e-5)


def test_v1_full_source_crop_maps_from_origin_without_translation():
    transform = ViewTransform(
        source_width=1280,
        source_height=960,
        crop_box=(0, 0, 1280, 960),
        scale=0.5,
        resized_width=640,
        resized_height=480,
        pad_x=0,
        pad_y=0,
        canvas_width=640,
        canvas_height=480,
    )

    assert transform.source_to_canvas([(0, 0), (1280, 960)]) == [(0.0, 0.0), (640.0, 480.0)]


def test_v7_crop_translation_is_included_in_coordinate_mapping():
    transform = ViewTransform(
        source_width=1600,
        source_height=1200,
        crop_box=(200, 100, 800, 600),
        scale=0.5,
        resized_width=400,
        resized_height=300,
        pad_x=10,
        pad_y=20,
        canvas_width=420,
        canvas_height=340,
    )

    assert transform.source_to_canvas([(200, 100), (1000, 700)]) == [(10.0, 20.0), (410.0, 320.0)]
    assert transform.canvas_to_source([(10, 20), (410, 320)]) == [(200.0, 100.0), (1000.0, 700.0)]


def test_json_compatible_dict_round_trip_has_stable_keys_and_values():
    transform = ViewTransform(
        source_width=1000,
        source_height=800,
        crop_box=(10, 20, 900, 700),
        scale=0.5,
        resized_width=450,
        resized_height=350,
        pad_x=4,
        pad_y=6,
        canvas_width=640,
        canvas_height=480,
    )

    serialized = transform.to_dict()

    assert serialized == transform.to_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert ViewTransform.from_dict(serialized) == transform
    assert ViewTransform.from_dict({
        "pair_id": "pair-42",
        "view": "v7",
        "instances": [],
        "transform": serialized,
    }) == transform


def test_effective_raster_scale_maps_bottom_edge_exactly_and_round_trips():
    transform = ViewTransform(
        source_width=1000,
        source_height=333,
        crop_box=(0, 0, 1000, 333),
        scale=0.64,
        resized_width=640,
        resized_height=213,
        pad_x=0,
        pad_y=17,
        canvas_width=640,
        canvas_height=640,
    )

    canvas_points = transform.source_to_canvas([(1000, 333)])

    assert canvas_points == [(640.0, 230.0)]
    assert transform.canvas_to_source(canvas_points) == pytest.approx([(1000.0, 333.0)], abs=1e-5)


def test_resized_dimensions_must_match_nominal_scale_flooring():
    with pytest.raises(ValueError, match="resized dimensions"):
        ViewTransform(
            source_width=1000,
            source_height=333,
            crop_box=(0, 0, 1000, 333),
            scale=0.64,
            resized_width=640,
            resized_height=214,
            pad_x=0,
            pad_y=0,
            canvas_width=640,
            canvas_height=640,
        )


def test_nominal_scale_must_produce_at_least_one_pixel_per_dimension():
    with pytest.raises(ValueError, match="resized dimensions"):
        ViewTransform(
            source_width=100,
            source_height=100,
            crop_box=(0, 0, 100, 100),
            scale=0.009,
            resized_width=1,
            resized_height=1,
            pad_x=0,
            pad_y=0,
            canvas_width=1,
            canvas_height=1,
        )


@pytest.mark.parametrize("kwargs", [
    {"crop_box": (-1, 0, 10, 10)},
    {"crop_box": (0, 0, 101, 10)},
    {"crop_box": (0, 0, 10, 101)},
    {"crop_box": (0, 0, 0, 10)},
    {"scale": 0},
    {"resized_width": 0},
    {"resized_height": -1},
    {"resized_width": 10.5},
    {"canvas_width": 0},
    {"canvas_height": -1},
    {"pad_x": -0.1},
    {"pad_x": 1, "resized_width": 10},
    {"pad_y": 1, "resized_height": 10},
    {"source_width": math.inf},
    {"crop_box": (0, math.nan, 10, 10)},
])
def test_invalid_dimensions_crop_and_nonfinite_values_raise_value_error(kwargs):
    values = {
        "source_width": 100,
        "source_height": 100,
        "crop_box": (0, 0, 10, 10),
        "scale": 1,
        "resized_width": 10,
        "resized_height": 10,
        "pad_x": 0,
        "pad_y": 0,
        "canvas_width": 10,
        "canvas_height": 10,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        ViewTransform(**values)


def test_point_generators_are_accepted_and_outputs_preserve_float_coordinates():
    transform = ViewTransform(
        source_width=100,
        source_height=100,
        crop_box=(0, 0, 100, 100),
        scale=0.5,
        resized_width=50,
        resized_height=50,
        pad_x=1.25,
        pad_y=2.5,
        canvas_width=52,
        canvas_height=55,
    )

    result = transform.source_to_canvas((point for point in [(1.5, 2.25), (3.75, 4.5)]))

    assert result == [(2.0, 3.625), (3.125, 4.75)]
    assert all(isinstance(value, float) for point in result for value in point)


def test_unsupported_schema_version_is_rejected():
    serialized = ViewTransform(
        source_width=100,
        source_height=100,
        crop_box=(0, 0, 100, 100),
        scale=1,
        resized_width=100,
        resized_height=100,
        pad_x=0,
        pad_y=0,
        canvas_width=100,
        canvas_height=100,
    ).to_dict()
    serialized["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        ViewTransform.from_dict(serialized)
