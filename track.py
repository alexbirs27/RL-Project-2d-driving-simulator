import math
from typing import List, Tuple


class Track:
    """
    Defines the racing track with road boundaries and finish line.

    The track is defined as a series of points forming the center line,
    with a specified width creating the road boundaries.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.road_width = 100

        self.center_points = self._create_oval_track()
        self.num_checkpoints = len(self.center_points)
        self.start_position = self.center_points[0]
        self.start_angle = self._calculate_start_angle()

        self.finish_line_start = 0
        self.finish_line_end = 1

    def _create_oval_track(self) -> List[Tuple[float, float]]:
        """Create an oval track with some curves."""
        points = []
        center_x = self.width / 2
        center_y = self.height / 2
        radius_x = self.width / 2 - 150
        radius_y = self.height / 2 - 100

        num_points = 100
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            wobble = 20 * math.sin(4 * angle)
            x = center_x + (radius_x + wobble) * math.cos(angle)
            y = center_y + (radius_y + wobble) * math.sin(angle)
            points.append((x, y))

        return points

    def _calculate_start_angle(self) -> float:
        """Calculate the starting angle based on track direction."""
        if len(self.center_points) < 2:
            return 0.0

        p1 = self.center_points[0]
        p2 = self.center_points[1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return math.atan2(dx, -dy)

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if a point is on the road."""
        min_dist = float('inf')

        for i in range(len(self.center_points)):
            p1 = self.center_points[i]
            p2 = self.center_points[(i + 1) % len(self.center_points)]

            dist = self._point_to_segment_distance(x, y, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist <= self.road_width / 2

    def _point_to_segment_distance(
        self,
        px: float,
        py: float,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """Calculate distance from point to line segment."""
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def check_finish_line(
        self,
        prev_x: float,
        prev_y: float,
        curr_x: float,
        curr_y: float
    ) -> bool:
        """Check if the car crossed the finish line."""
        p1 = self.center_points[self.finish_line_start]
        p2 = self.center_points[self.finish_line_end]

        direction = (p2[0] - p1[0], p2[1] - p1[1])
        perpendicular = (-direction[1], direction[0])
        length = math.sqrt(perpendicular[0] ** 2 + perpendicular[1] ** 2)
        if length == 0:
            return False
        perpendicular = (
            perpendicular[0] / length * self.road_width / 2,
            perpendicular[1] / length * self.road_width / 2
        )

        line_start = (p1[0] + perpendicular[0], p1[1] + perpendicular[1])
        line_end = (p1[0] - perpendicular[0], p1[1] - perpendicular[1])

        return self._lines_intersect(
            prev_x, prev_y, curr_x, curr_y,
            line_start[0], line_start[1],
            line_end[0], line_end[1]
        )

    def _lines_intersect(
        self,
        x1: float, y1: float, x2: float, y2: float,
        x3: float, y3: float, x4: float, y4: float
    ) -> bool:
        """Check if two line segments intersect."""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1

    def get_finish_line_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the start and end points of the finish line for rendering."""
        p1 = self.center_points[self.finish_line_start]
        p2 = self.center_points[self.finish_line_end]

        direction = (p2[0] - p1[0], p2[1] - p1[1])
        perpendicular = (-direction[1], direction[0])
        length = math.sqrt(perpendicular[0] ** 2 + perpendicular[1] ** 2)
        if length == 0:
            return (p1, p1)
        perpendicular = (
            perpendicular[0] / length * self.road_width / 2,
            perpendicular[1] / length * self.road_width / 2
        )

        return (
            (p1[0] + perpendicular[0], p1[1] + perpendicular[1]),
            (p1[0] - perpendicular[0], p1[1] - perpendicular[1])
        )

    def get_nearest_checkpoint(self, x: float, y: float) -> int:
        """Get the index of the nearest checkpoint to the given position."""
        min_dist = float('inf')
        nearest_idx = 0

        for i, point in enumerate(self.center_points):
            dist = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def get_progress(self, checkpoint: int) -> float:
        """Get progress as a fraction of the track completed (0.0 to 1.0)."""
        return checkpoint / self.num_checkpoints
