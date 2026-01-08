"""
Load F1Tenth racetrack from PNG image and YAML metadata.
"""
import numpy as np
from PIL import Image
import yaml
from typing import Tuple, List


class ImageTrack:
    """Track loaded from PNG occupancy grid map."""

    def __init__(self, png_path: str, yaml_path: str, road_width: float = 100.0):
        """
        Load track from F1Tenth format files.

        Args:
            png_path: Path to PNG occupancy grid
            yaml_path: Path to YAML metadata file
            road_width: Track width in pixels (for visualization)
        """
        # Load PNG as grayscale
        self.image = Image.open(png_path).convert('L')
        self.map_array = np.array(self.image)

        # Load YAML metadata
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

        self.resolution = metadata['resolution']  # meters per pixel
        self.origin = metadata['origin']  # [x, y, theta] in meters

        self.height, self.width = self.map_array.shape
        self.road_width = road_width

        # Find start position (first white pixel from top-left)
        self.start_position = self._find_start_position()
        self.start_angle = 0.0  # Default, can be improved

        # Create dummy checkpoints along track centerline
        self.center_points = self._extract_centerline()
        self.num_checkpoints = len(self.center_points)

        # No obstacles for now
        self.obstacles = []

        print(f"Loaded track: {self.width}x{self.height} pixels")
        print(f"Resolution: {self.resolution}m/pixel")
        print(f"Start position: {self.start_position}")
        print(f"Checkpoints: {self.num_checkpoints}")

    def _find_start_position(self) -> Tuple[float, float]:
        """Find a suitable start position on the track."""
        # Find first white pixel (track) from center
        center_y = self.height // 2
        center_x = self.width // 2

        # Search around center for track
        for radius in range(10, min(self.width, self.height) // 2, 10):
            for angle in np.linspace(0, 2 * np.pi, 36):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.map_array[y, x] > 250:  # White = track
                        return (float(x), float(y))

        # Fallback to center
        return (float(center_x), float(center_y))

    def _extract_centerline(self) -> List[Tuple[float, float]]:
        """Extract centerline points from track (simplified version)."""
        # For now, create a grid of checkpoints on white pixels
        checkpoints = []
        step = 20  # Checkpoint every 20 pixels

        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                if self.map_array[y, x] > 250:  # White = track
                    checkpoints.append((float(x), float(y)))

        if not checkpoints:
            # Fallback: single checkpoint at start
            checkpoints = [self.start_position]

        return checkpoints

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if position (x, y) is on the track."""
        ix = int(x)
        iy = int(y)

        # Out of bounds = off road
        if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
            return False

        # White pixels (>250) = track, black/gray = off-road
        return self.map_array[iy, ix] > 250

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
        # Use start position with perpendicular line
        x, y = self.start_position
        # Simple horizontal line for now
        return ((x - 20, y), (x + 20, y))

    def check_finish_line(self, prev_x: float, prev_y: float,
                         curr_x: float, curr_y: float) -> bool:
        """Check if car crossed finish line."""
        # Simple distance-based check for now
        start_x, start_y = self.start_position
        dist = ((curr_x - start_x) ** 2 + (curr_y - start_y) ** 2) ** 0.5
        return dist < 30  # Within 30 pixels of start

    def check_obstacle_collision(self, x: float, y: float, car_radius: float = 15) -> bool:
        """Check collision with obstacles."""
        return False  # No obstacles for now

    def get_road_width_at_segment(self, segment_idx: int) -> float:
        """Get road width at segment (constant for now)."""
        return self.road_width
