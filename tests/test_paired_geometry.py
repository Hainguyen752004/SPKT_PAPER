import math

import pytest

from data_processing.paired_geometry import (
    EPSILON,
    clip_polygon_to_rect,
    deduplicate_polygon,
    polygon_area,
    polygon_intersects_rect,
)


def assert_points_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected):
        assert actual_point == pytest.approx(expected_point)


def test_polygon_fully_inside_preserves_vertices_and_area():
    polygon = [(0.25, 0.25), (1.75, 0.25), (1.0, 1.5)]

    clipped = clip_polygon_to_rect(polygon, 0.0, 0.0, 2.0, 2.0)

    assert_points_close(clipped, polygon)
    assert polygon_area(clipped) == pytest.approx(polygon_area(polygon))
    assert all(isinstance(value, float) for point in clipped for value in point)


def test_polygon_fully_outside_returns_empty_polygon():
    assert clip_polygon_to_rect([(-3, -3), (-2, -3), (-2, -2)], 0, 0, 1, 1) == []


def test_polygon_crossing_all_four_edges_is_clipped_to_rectangle():
    clipped = clip_polygon_to_rect([(-1, -1), (2, -1), (2, 2), (-1, 2)], 0, 0, 1, 1)

    assert_points_close(clipped, [(0, 1), (0, 0), (1, 0), (1, 1)])
    assert polygon_area(clipped) == pytest.approx(1.0)


def test_adjacent_and_closing_duplicates_are_removed():
    polygon = [(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)]

    assert_points_close(deduplicate_polygon(polygon), [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert_points_close(clip_polygon_to_rect(polygon, 0, 0, 1, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.mark.parametrize("polygon", [
    [(0, 0), (1, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(-1, 0), (0, 0), (1, 0)],
])
def test_degenerate_output_returns_empty_polygon(polygon):
    assert clip_polygon_to_rect(polygon, 0, 0, 1, 1) == []


@pytest.mark.parametrize("rectangle", [
    (1, 0, 0, 1),
    (0, 1, 1, 0),
])
def test_invalid_rectangle_raises_value_error(rectangle):
    with pytest.raises(ValueError, match="rectangle"):
        clip_polygon_to_rect([(0, 0), (1, 0), (0, 1)], *rectangle)


def test_parallel_boundary_segments_do_not_divide_by_zero():
    polygon = [(0, 0), (2, 0), (2, 1), (0, 1)]

    clipped = clip_polygon_to_rect(polygon, 0, 0, 1, 1)

    assert_points_close(clipped, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert math.isfinite(polygon_area(clipped))


def test_near_boundary_transition_clips_to_exact_boundary_without_leakage():
    polygon = [(-1.1e-9, 0), (-1e-9, 1), (1, 1), (1, 0)]

    clipped = clip_polygon_to_rect(polygon, 0, 0, 1, 1)

    assert_points_close(clipped, [(0, 0), (0, 1), (1, 1), (1, 0)])
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in clipped)


def test_disconnected_concave_intersection_returns_empty_polygon():
    # The rectangle intersects the two uprights of this U shape, but not its middle.
    u_shape = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]

    assert clip_polygon_to_rect(u_shape, 0, 2, 4, 3) == []


def test_polygon_intersection_rejects_diagonal_polygon_with_overlapping_bbox():
    triangle = [(0, 0), (4, 0), (0, 4)]

    assert polygon_intersects_rect(triangle, 3, 3, 4, 4) is False


def test_polygon_intersection_counts_boundary_contact_and_concave_crossing():
    triangle = [(0, 0), (4, 0), (0, 4)]
    u_shape = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]

    assert polygon_intersects_rect(triangle, 2, 2, 3, 3) is True
    assert polygon_intersects_rect(u_shape, 0, 2, 4, 3) is True


@pytest.mark.parametrize("rectangle", [
    (0, 0, 0, 1),
    (0, 1, 1, 0),
    (float("nan"), 0, 1, 1),
    (0, 0, float("inf"), 1),
])
def test_polygon_intersection_validates_rectangle_like_clipping(rectangle):
    with pytest.raises(ValueError, match="rectangle bounds"):
        polygon_intersects_rect([(0, 0), (1, 0), (0, 1)], *rectangle)


@pytest.mark.parametrize("rectangle", [
    (float("nan"), 0, 1, 1),
    (0, float("nan"), 1, 1),
    (0, 0, float("inf"), 1),
    (0, 0, 1, float("-inf")),
])
def test_non_finite_rectangle_bounds_raise_value_error(rectangle):
    with pytest.raises(ValueError, match="rectangle"):
        clip_polygon_to_rect([(0, 0), (1, 0), (0, 1)], *rectangle)


def test_polygon_area_is_orientation_independent_and_zero_for_fewer_than_three_points():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]

    assert polygon_area(square) == pytest.approx(1.0)
    assert polygon_area(list(reversed(square))) == pytest.approx(1.0)
    assert polygon_area([(0, 0), (1, 0)]) == 0.0
    assert EPSILON == 1e-9
