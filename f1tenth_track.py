"""
Load F1Tenth track from centerline CSV - creates clean procedural track.
"""
import csv
import numpy as np
from typing import List, Tuple


class F1TenthTrack:
    """Clean track loaded from F1Tenth centerline CSV."""

    def __init__(self, csv_path: str, scale: float = 50.0, road_width: float = 100.0):
        """
        Load track from centerline CSV and create clean procedural track.

        Args:
            csv_path: Path to centerline CSV file
            scale: Pixels per meter (higher = bigger track)
            road_width: Track width in pixels
        """
        # Load centerline points from CSV
        # Format: # x_m, y_m, w_tr_right_m, w_tr_left_m
        raw_points = []
        track_widths = []  # Store actual track width at each point
        with open(csv_path, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue

                # Parse CSV: x_m, y_m, w_tr_right_m, w_tr_left_m
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    x = float(parts[0].strip()) * scale
                    y = float(parts[1].strip()) * scale
                    w_right = float(parts[2].strip()) * scale
                    w_left = float(parts[3].strip()) * scale
                    total_width = (w_right + w_left)
                    raw_points.append((x, y))
                    track_widths.append(total_width)

        # Offset to positive coordinates and center in a reasonable space
        if raw_points:
            min_x = min(p[0] for p in raw_points)
            min_y = min(p[1] for p in raw_points)
            max_x = max(p[0] for p in raw_points)
            max_y = max(p[1] for p in raw_points)

            # Center the track
            offset_x = -min_x + 200
            offset_y = -min_y + 200

            self.center_points = [(x + offset_x, y + offset_y) for x, y in raw_points]
            self.track_widths = track_widths  # Actual track widths from CSV

            # Create a subsampled version for fast collision detection (every 5th point)
            self.collision_points = [(self.center_points[i], self.track_widths[i])
                                     for i in range(0, len(self.center_points), 5)]

            self.width = int(max_x - min_x + 400)
            self.height = int(max_y - min_y + 400)
        else:
            self.center_points = [(500, 500)]
            self.track_widths = [road_width]
            self.collision_points = [((500, 500), road_width)]
            self.width = 1000
            self.height = 1000

        self.road_width = road_width  # Default for rendering
        self.num_checkpoints = len(self.center_points)

        # Start at first point
        if self.center_points:
            self.start_position = self.center_points[0]
        else:
            self.start_position = (500, 500)

        # Calculate start angle from first two points
        if len(self.center_points) >= 2:
            dx = self.center_points[1][0] - self.center_points[0][0]
            dy = self.center_points[1][1] - self.center_points[0][1]
            import math
            self.start_angle = math.atan2(dx, -dy)
        else:
            self.start_angle = 0.0

        # No obstacles
        self.obstacles = []

        print(f"Loaded F1Tenth track from CSV")
        print(f"Track size: {self.width}x{self.height}")
        print(f"Centerline points: {self.num_checkpoints}")
        print(f"Start position: {self.start_position}")

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if position is on the track (fast subsampled collision detection)."""
        # Find nearest collision point (subsampled for speed)
        min_dist = float('inf')
        nearest_width = self.road_width

        for (cx, cy), width in self.collision_points:
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_width = width

        # On road if within track width (slightly generous to avoid false positives)
        return min_dist <= (nearest_width / 2 * 1.1)

    def get_nearest_checkpoint(self, x: float, y: float) -> int:
        """Find nearest checkpoint index."""
        min_dist = float('inf')
        nearest_idx = 0

        for i, (cx, cy) in enumerate(self.center_points):
            dist = (x - cx) ** 2 + (y - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def get_progress(self, checkpoint_idx: int) -> float:
        """Get progress around track (0.0 to 1.0)."""
        return checkpoint_idx / max(1, self.num_checkpoints)

    def get_finish_line_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get finish line endpoints."""
        import math
        if len(self.center_points) >= 2:
            x, y = self.center_points[0]
            x2, y2 = self.center_points[1]

            # Perpendicular to track direction
            dx = x2 - x
            dy = y2 - y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_x = -dy / length * (self.road_width / 2)
                perp_y = dx / length * (self.road_width / 2)

                return ((x + perp_x, y + perp_y), (x - perp_x, y - perp_y))

        # Fallback
        x, y = self.start_position
        return ((x - 30, y), (x + 30, y))

    def check_finish_line(self, prev_x: float, prev_y: float,
                         curr_x: float, curr_y: float) -> bool:
        """Check if car crossed finish line."""
        # Simple distance-based check
        start_x, start_y = self.start_position
        dist = ((curr_x - start_x) ** 2 + (curr_y - start_y) ** 2) ** 0.5
        return dist < 50  # Within 50 pixels of start

    def check_obstacle_collision(self, x: float, y: float, car_radius: float = 15) -> bool:
        """Check collision with obstacles."""
        return False  # No obstacles

    def get_road_width_at_segment(self, segment_idx: int) -> float:
        """Get road width at segment (from CSV data)."""
        if segment_idx < len(self.track_widths):
            return self.track_widths[segment_idx]
        return self.road_width
