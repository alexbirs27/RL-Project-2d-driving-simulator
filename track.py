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
        self.road_width = 100  # Default width

        # Create complex track
        self.center_points = self._create_complex_track()
        self.num_checkpoints = len(self.center_points)

        # Variable width sections (segment_index: width)
        self.width_sections = self._create_width_sections()

        # Obstacles (x, y, radius)
        self.obstacles = self._create_obstacles()

        # Start at point 10 where finish line is (perfectly horizontal)
        self.start_position = self.center_points[10]
        self.start_angle = math.pi / 2  # Pointing right (horizontal)

        # Finish line in middle of start straight (perfectly horizontal section)
        self.finish_line_start = 10
        self.finish_line_end = 11

    def _create_complex_track(self) -> List[Tuple[float, float]]:
        """Create a simple rounded rectangle track - clean and smooth."""
        points = []

        # Track dimensions
        center_x = 800
        center_y = 700
        width = 500   # Half-width of straight sections
        height = 350  # Half-height of straight sections
        corner_radius = 150

        num_points_straight = 15
        num_points_corner = 18

        # START on bottom straight (left to right) - main straight
        # Extra long to ensure perfect horizontal start
        for i in range(25):
            x = center_x - width + (i * (2 * width) / 25)
            y = center_y + height
            points.append((x, y))

        # Bottom-right corner
        for i in range(1, num_points_corner):
            angle = (math.pi / 2) * (i / num_points_corner)
            x = center_x + width + corner_radius * (1 - math.cos(angle))
            y = center_y + height - corner_radius * math.sin(angle)
            points.append((x, y))

        # Right straight (bottom to top)
        for i in range(num_points_straight):
            x = center_x + width + corner_radius
            y = center_y + height - corner_radius - (i * (2 * height - 2 * corner_radius) / num_points_straight)
            points.append((x, y))

        # Top-right corner
        for i in range(1, num_points_corner):
            angle = (math.pi / 2) * (i / num_points_corner)
            x = center_x + width + corner_radius - corner_radius * math.sin(angle)
            y = center_y - height + corner_radius - corner_radius * (1 - math.cos(angle))
            points.append((x, y))

        # Top straight (right to left)
        for i in range(num_points_straight):
            x = center_x + width - (i * (2 * width) / num_points_straight)
            y = center_y - height
            points.append((x, y))

        # Top-left corner
        for i in range(1, num_points_corner):
            angle = (math.pi / 2) * (i / num_points_corner)
            x = center_x - width - corner_radius * (1 - math.cos(angle))
            y = center_y - height + corner_radius * math.sin(angle)
            points.append((x, y))

        # Left straight (top to bottom)
        for i in range(num_points_straight):
            x = center_x - width - corner_radius
            y = center_y - height + corner_radius + (i * (2 * height - 2 * corner_radius) / num_points_straight)
            points.append((x, y))

        # Bottom-left corner
        for i in range(1, num_points_corner):
            angle = (math.pi / 2) * (i / num_points_corner)
            x = center_x - width - corner_radius + corner_radius * math.sin(angle)
            y = center_y + height - corner_radius + corner_radius * (1 - math.cos(angle))
            points.append((x, y))

        return points

    def _create_width_sections(self) -> dict:
        """Define narrow sections of the track."""
        # All corners same width - clean and simple
        return {}

    def _create_obstacles(self) -> List[Tuple[float, float, float]]:
        """Create obstacles (cones/barriers) placed strategically on track."""
        # No obstacles - keep it simple and clean
        return []

    def _calculate_start_angle(self) -> float:
        """Calculate the starting angle based on track direction."""
        if len(self.center_points) < 2:
            return 0.0

        p1 = self.center_points[0]
        p2 = self.center_points[1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return math.atan2(dx, -dy)

    def get_road_width_at_segment(self, segment_idx: int) -> float:
        """Get the road width at a specific segment (variable width)."""
        return self.width_sections.get(segment_idx, self.road_width)

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if a point is on the road (accounts for variable width)."""
        min_dist = float('inf')
        closest_segment = 0

        for i in range(len(self.center_points)):
            p1 = self.center_points[i]
            p2 = self.center_points[(i + 1) % len(self.center_points)]

            dist = self._point_to_segment_distance(x, y, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_segment = i

        # Get width for the closest segment
        segment_width = self.get_road_width_at_segment(closest_segment)
        return min_dist <= segment_width / 2

    def check_obstacle_collision(self, x: float, y: float, car_radius: float = 15) -> bool:
        """Check if car collides with any obstacle."""
        for obs_x, obs_y, obs_radius in self.obstacles:
            dist = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
            if dist < (car_radius + obs_radius):
                return True
        return False

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
        """Check if the car crossed the finish line in the forward direction."""
        p1 = self.center_points[self.finish_line_start]
        p2 = self.center_points[self.finish_line_end]

        # Track direction at finish line (forward direction)
        track_direction = (p2[0] - p1[0], p2[1] - p1[1])

        # Car movement direction
        car_direction = (curr_x - prev_x, curr_y - prev_y)

        # Check if car is moving in the forward direction (dot product > 0)
        dot_product = track_direction[0] * car_direction[0] + track_direction[1] * car_direction[1]
        if dot_product <= 0:
            # Car is moving backwards - don't count as lap completion
            return False

        perpendicular = (-track_direction[1], track_direction[0])
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
