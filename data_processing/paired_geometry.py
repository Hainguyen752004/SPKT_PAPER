"""Small, dependency-free helpers for paired segmentation geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

EPSILON = 1e-9


@dataclass(frozen=True)
class ViewTransform:
    """Affine mapping between source-image and padded canvas coordinates.

    ``crop_box`` is expressed as ``(x, y, width, height)`` in source pixels.
    ``scale`` records the nominal preprocessing scale; the resized dimensions
    must match ``int(crop_dimension * scale)``. Coordinate conversion instead
    uses effective per-axis scales derived from those integer raster dimensions,
    so it matches the pixels produced by image resizing exactly.
    When ``canvas_height`` is omitted, the canvas is square with the supplied
    ``canvas_width``.
    """

    source_width: float
    source_height: float
    crop_box: tuple[float, float, float, float]
    scale: float
    resized_width: int
    resized_height: int
    pad_x: float
    pad_y: float
    canvas_width: float
    canvas_height: float | None = None

    def __post_init__(self):
        source_width = _finite_float(self.source_width, "source_width", positive=True)
        source_height = _finite_float(self.source_height, "source_height", positive=True)
        scale = _finite_float(self.scale, "scale", positive=True)
        resized_width = _positive_integer(self.resized_width, "resized_width")
        resized_height = _positive_integer(self.resized_height, "resized_height")
        pad_x = _finite_float(self.pad_x, "pad_x", nonnegative=True)
        pad_y = _finite_float(self.pad_y, "pad_y", nonnegative=True)
        canvas_width = _finite_float(self.canvas_width, "canvas_width", positive=True)
        canvas_height = canvas_width if self.canvas_height is None else _finite_float(
            self.canvas_height, "canvas_height", positive=True
        )

        try:
            crop_x, crop_y, crop_width, crop_height = self.crop_box
        except (TypeError, ValueError) as error:
            raise ValueError("crop_box must contain (x, y, width, height)") from error
        crop_x = _finite_float(crop_x, "crop_box x", nonnegative=True)
        crop_y = _finite_float(crop_y, "crop_box y", nonnegative=True)
        crop_width = _finite_float(crop_width, "crop_box width", positive=True)
        crop_height = _finite_float(crop_height, "crop_box height", positive=True)
        if crop_x + crop_width > source_width or crop_y + crop_height > source_height:
            raise ValueError("crop_box must be fully within source bounds")
        expected_resized_width = int(crop_width * scale)
        expected_resized_height = int(crop_height * scale)
        if expected_resized_width < 1 or expected_resized_height < 1:
            raise ValueError("nominal scale must produce resized dimensions of at least one pixel")
        if (resized_width, resized_height) != (expected_resized_width, expected_resized_height):
            raise ValueError("resized dimensions must match nominal scale flooring")
        if pad_x + resized_width > canvas_width or pad_y + resized_height > canvas_height:
            raise ValueError("resized dimensions plus padding must fit within canvas bounds")

        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "source_height", source_height)
        object.__setattr__(self, "crop_box", (crop_x, crop_y, crop_width, crop_height))
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "resized_width", resized_width)
        object.__setattr__(self, "resized_height", resized_height)
        object.__setattr__(self, "pad_x", pad_x)
        object.__setattr__(self, "pad_y", pad_y)
        object.__setattr__(self, "canvas_width", canvas_width)
        object.__setattr__(self, "canvas_height", canvas_height)

    def source_to_canvas(self, points):
        """Map an iterable of source-coordinate points to canvas coordinates."""
        crop_x, crop_y, crop_width, crop_height = self.crop_box
        scale_x = self.resized_width / crop_width
        scale_y = self.resized_height / crop_height
        return [
            ((x - crop_x) * scale_x + self.pad_x, (y - crop_y) * scale_y + self.pad_y)
            for x, y in (_coordinate_pair(point) for point in points)
        ]

    def canvas_to_source(self, points):
        """Map an iterable of canvas-coordinate points to source coordinates."""
        crop_x, crop_y, crop_width, crop_height = self.crop_box
        scale_x = self.resized_width / crop_width
        scale_y = self.resized_height / crop_height
        return [
            ((x - self.pad_x) / scale_x + crop_x, (y - self.pad_y) / scale_y + crop_y)
            for x, y in (_coordinate_pair(point) for point in points)
        ]

    def to_dict(self):
        """Return a deterministic, JSON-compatible representation."""
        return {
            "schema_version": 1,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "crop_box": list(self.crop_box),
            "scale": self.scale,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
        }

    @classmethod
    def from_dict(cls, values):
        """Construct a transform from a v1 transform dict or containing record."""
        record = dict(values)
        transform = dict(record["transform"]) if "transform" in record else record
        if transform.get("schema_version") != 1:
            raise ValueError("unsupported transform schema version")
        field_names = (
            "source_width",
            "source_height",
            "crop_box",
            "scale",
            "resized_width",
            "resized_height",
            "pad_x",
            "pad_y",
            "canvas_width",
            "canvas_height",
        )
        return cls(**{name: transform[name] for name in field_names if name in transform})


def _finite_float(value, name, positive=False, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _coordinate_pair(point):
    try:
        x, y = point
    except (TypeError, ValueError) as error:
        raise ValueError("points must contain (x, y) coordinate pairs") from error
    return _finite_float(x, "point x"), _finite_float(y, "point y")


def polygon_area(points):
    """Return the unsigned shoelace area of a polygon."""
    vertices = list(points)
    if len(vertices) < 3:
        return 0.0
    return abs(sum(
        float(vertices[index][0]) * float(vertices[(index + 1) % len(vertices)][1])
        - float(vertices[(index + 1) % len(vertices)][0]) * float(vertices[index][1])
        for index in range(len(vertices))
    )) / 2.0


def deduplicate_polygon(points, epsilon=EPSILON):
    """Remove adjacent near-identical vertices and a duplicate closing vertex."""
    result = []
    for point in points:
        vertex = (float(point[0]), float(point[1]))
        if not result or not _points_close(result[-1], vertex, epsilon):
            result.append(vertex)
    if len(result) > 1 and _points_close(result[0], result[-1], epsilon):
        result.pop()
    return result


def clip_polygon_to_rect(points, x_min, y_min, x_max, y_max):
    """Clip a polygon against an axis-aligned rectangle using Sutherland-Hodgman.

    The phase-1 API represents one polygon only.  It returns ``[]`` when clipping
    would create a disconnected, self-touching, or self-intersecting result rather
    than joining separate components with an artificial rectangle-boundary bridge.
    """
    x_min, y_min, x_max, y_max = _validated_rectangle(x_min, y_min, x_max, y_max)

    source = deduplicate_polygon(points)
    clipped = source
    if len(clipped) < 3:
        return []

    boundaries = (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    )
    for axis, boundary, keep_greater in boundaries:
        clipped = _clip_against_boundary(clipped, axis, boundary, keep_greater)
        clipped = deduplicate_polygon(clipped)
        if len(clipped) < 3:
            return []

    if (polygon_area(clipped) <= EPSILON
            or _has_non_adjacent_duplicates(clipped)
            or _has_self_intersections(clipped)
            or _has_artificial_boundary_bridge(clipped, source, x_min, y_min, x_max, y_max)):
        return []
    return clipped


def polygon_intersects_rect(points, x_min, y_min, x_max, y_max):
    """Return whether a polygon and rectangle have any interior or boundary contact."""
    x_min, y_min, x_max, y_max = _validated_rectangle(x_min, y_min, x_max, y_max)
    source = deduplicate_polygon(_coordinate_pair(point) for point in points)
    if len(source) < 3:
        return False

    def inside_rectangle(point):
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

    if any(inside_rectangle(point) for point in source):
        return True

    corners = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    if any(_point_in_or_on_polygon(corner, source) for corner in corners):
        return True

    rectangle_edges = tuple(zip(corners, corners[1:] + corners[:1]))
    polygon_edges = tuple(zip(source, source[1:] + source[:1]))
    return any(
        _segments_intersect(polygon_start, polygon_end, rect_start, rect_end)
        for polygon_start, polygon_end in polygon_edges
        for rect_start, rect_end in rectangle_edges
    )


def _validated_rectangle(x_min, y_min, x_max, y_max):
    bounds = tuple(map(float, (x_min, y_min, x_max, y_max)))
    if not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("rectangle bounds must satisfy x_min < x_max and y_min < y_max")
    return bounds


def _points_close(first, second, epsilon):
    return abs(first[0] - second[0]) <= epsilon and abs(first[1] - second[1]) <= epsilon


def _clip_against_boundary(points, axis, boundary, keep_greater):
    if not points:
        return []

    def is_inside(point):
        value = point[axis]
        return value >= boundary if keep_greater else value <= boundary

    def intersection(start, end):
        delta = end[axis] - start[axis]
        if delta == 0.0:
            return (boundary, float(start[1])) if axis == 0 else (float(start[0]), boundary)
        ratio = (boundary - start[axis]) / delta
        other_axis = 1 - axis
        other_value = float(start[other_axis] + ratio * (end[other_axis] - start[other_axis]))
        return (boundary, other_value) if axis == 0 else (other_value, boundary)

    output = []
    previous = points[-1]
    previous_inside = is_inside(previous)
    for current in points:
        current_inside = is_inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous, previous_inside = current, current_inside
    return output


def _has_non_adjacent_duplicates(points):
    return any(
        _points_close(points[index], points[other_index], EPSILON)
        for index in range(len(points))
        for other_index in range(index + 1, len(points))
    )


def _has_self_intersections(points):
    count = len(points)
    for first_index in range(count):
        first_end = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_end = (second_index + 1) % count
            if first_index in (second_index, second_end) or first_end in (second_index, second_end):
                continue
            if _segments_intersect(points[first_index], points[first_end], points[second_index], points[second_end]):
                return True
    return False


def _segments_intersect(first_start, first_end, second_start, second_end):
    def orientation(start, end, point):
        return ((end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0]))

    def on_segment(start, end, point):
        return (abs(orientation(start, end, point)) <= EPSILON
                and min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON
                and min(start[1], end[1]) - EPSILON <= point[1] <= max(start[1], end[1]) + EPSILON)

    first_a = orientation(first_start, first_end, second_start)
    first_b = orientation(first_start, first_end, second_end)
    second_a = orientation(second_start, second_end, first_start)
    second_b = orientation(second_start, second_end, first_end)
    if abs(first_a) <= EPSILON and on_segment(first_start, first_end, second_start):
        return True
    if abs(first_b) <= EPSILON and on_segment(first_start, first_end, second_end):
        return True
    if abs(second_a) <= EPSILON and on_segment(second_start, second_end, first_start):
        return True
    if abs(second_b) <= EPSILON and on_segment(second_start, second_end, first_end):
        return True
    return (first_a > EPSILON) != (first_b > EPSILON) and (second_a > EPSILON) != (second_b > EPSILON)


def _has_artificial_boundary_bridge(clipped, source, x_min, y_min, x_max, y_max):
    source = deduplicate_polygon(source)
    if len(source) < 3:
        return False
    for index, start in enumerate(clipped):
        end = clipped[(index + 1) % len(clipped)]
        on_rectangle_boundary = (
            (abs(start[0] - x_min) <= EPSILON and abs(end[0] - x_min) <= EPSILON)
            or (abs(start[0] - x_max) <= EPSILON and abs(end[0] - x_max) <= EPSILON)
            or (abs(start[1] - y_min) <= EPSILON and abs(end[1] - y_min) <= EPSILON)
            or (abs(start[1] - y_max) <= EPSILON and abs(end[1] - y_max) <= EPSILON)
        )
        if on_rectangle_boundary:
            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            if not _point_in_or_on_polygon(midpoint, source):
                return True
    return False


def _point_in_or_on_polygon(point, polygon):
    def point_on_segment(start, end):
        cross = ((end[0] - start[0]) * (point[1] - start[1])
                 - (end[1] - start[1]) * (point[0] - start[0]))
        return (abs(cross) <= EPSILON
                and min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON
                and min(start[1], end[1]) - EPSILON <= point[1] <= max(start[1], end[1]) + EPSILON)

    inside = False
    previous = polygon[-1]
    for current in polygon:
        if point_on_segment(previous, current):
            return True
        if ((current[1] > point[1]) != (previous[1] > point[1])):
            crossing_x = ((previous[0] - current[0]) * (point[1] - current[1])
                          / (previous[1] - current[1]) + current[0])
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside
