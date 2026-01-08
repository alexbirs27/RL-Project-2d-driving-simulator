import pygame
from typing import Optional

from car import Car
from track import Track
from config import RAY_MAX_DISTANCE, RAY_COLORS


class Renderer:
    """
    Handles all rendering operations.

    Can be disabled entirely for headless simulation during RL training.
    """

    GRASS_COLOR = (34, 139, 34)
    ROAD_COLOR = (60, 60, 60)
    ROAD_EDGE_COLOR = (255, 255, 255)
    CAR_COLOR = (255, 0, 0)  # Bright red - easy to see
    FINISH_LINE_COLOR = (255, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    SKID_MARK_COLOR = (20, 20, 20)  # Dark tire marks
    OBSTACLE_COLOR = (255, 100, 0)  # Orange cones/barriers
    NARROW_SECTION_COLOR = (255, 200, 0)  # Yellow marking for narrow sections
    CHECKPOINT_COLOR = (100, 100, 255)  # Blue for checkpoints
    NEXT_CHECKPOINT_COLOR = (0, 255, 0)  # Green for NEXT checkpoint
    VISITED_CHECKPOINT_COLOR = (150, 150, 150)  # Gray for visited

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.enabled = True
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0  # Zoom level
        self.show_debug = True  # Always show rays (toggle with 'D' key)

    def init(self):
        """Initialize pygame display."""
        if not self.enabled:
            return

        pygame.display.set_caption("2D Racing Engine - Press D for debug view")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)

    def _world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates with camera offset and zoom."""
        screen_x = (x - self.camera_x) * self.zoom
        screen_y = (y - self.camera_y) * self.zoom
        return (int(screen_x), int(screen_y))

    def render(
        self,
        car: Car,
        track: Track,
        lap_time: float,
        lap_complete: bool,
        best_time: Optional[float],
        next_checkpoint: int = None,
        visited_checkpoints: set = None
    ):
        """Render the complete game state."""
        if not self.enabled or self.screen is None:
            return

        # Camera is set externally (supports zoom)
        # Don't recalculate here

        self.screen.fill(self.GRASS_COLOR)
        self._draw_track(track)
        self._draw_checkpoints(track, next_checkpoint, visited_checkpoints)  # Draw checkpoints
        self._draw_obstacles(track)  # Draw obstacles
        self._draw_finish_line(track)
        self._draw_skid_marks(car)  # Draw skid marks before car
        self._draw_car(car)
        if self.show_debug:
            self._draw_debug_observations(car, track)
        self._draw_ui(lap_time, lap_complete, best_time, car)
        self._draw_minimap(car, track)  # Draw minimap last (on top)

        pygame.display.flip()

    def _draw_track(self, track):
        """Draw the road surface (supports both procedural and image-based tracks)."""
        if self.screen is None:
            return

        # Check if this is an ImageTrack (has map_array attribute)
        if hasattr(track, 'map_array'):
            # Draw PNG track image
            import numpy as np
            from PIL import Image

            # Get visible portion of track (with camera offset and zoom)
            left = int(self.camera_x)
            top = int(self.camera_y)
            right = left + int(self.width / self.zoom)
            bottom = top + int(self.height / self.zoom)

            # Clamp to track bounds
            left = max(0, left)
            top = max(0, top)
            right = min(track.width, right)
            bottom = min(track.height, bottom)

            # Extract visible region
            if right > left and bottom > top:
                visible_array = track.map_array[top:bottom, left:right]

                # Convert to RGB with proper colors:
                # White pixels (>250) = track (gray), else = grass (green)
                rgb_array = np.zeros((visible_array.shape[0], visible_array.shape[1], 3), dtype=np.uint8)

                # Create mask for track (white pixels)
                track_mask = visible_array > 250

                # Track = asphalt gray (lighter)
                rgb_array[track_mask] = [80, 80, 80]

                # Grass = darker green
                rgb_array[~track_mask] = [20, 100, 20]

                # Add track edges using numpy (faster)
                # Dilate and erode to find edges
                from scipy.ndimage import binary_dilation, binary_erosion
                try:
                    dilated = binary_dilation(track_mask)
                    eroded = binary_erosion(track_mask)
                    edges = dilated & ~eroded
                    rgb_array[edges] = [255, 255, 255]  # White edges
                except:
                    pass  # Skip edges if scipy not available

                # Convert to pygame surface
                track_surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))

                # Scale surface if zoomed
                if self.zoom != 1.0:
                    new_width = int(track_surface.get_width() * self.zoom)
                    new_height = int(track_surface.get_height() * self.zoom)
                    track_surface = pygame.transform.scale(track_surface, (new_width, new_height))

                # Draw at origin (camera offset already applied)
                screen_x = 0
                screen_y = 0
                self.screen.blit(track_surface, (screen_x, screen_y))

        else:
            # Draw procedural track (original code)
            # Draw track segments with variable width
            for i in range(len(track.center_points)):
                p1 = track.center_points[i]
                p2 = track.center_points[(i + 1) % len(track.center_points)]

                # Get width for this segment
                segment_width = track.get_road_width_at_segment(i)

                # Different color for narrow sections
                color = self.ROAD_COLOR
                if segment_width < track.road_width:
                    # Blend yellow tint for narrow sections
                    color = (80, 80, 50)  # Darker yellowish for narrow parts

                # Apply camera offset
                screen_p1 = self._world_to_screen(p1[0], p1[1])
                screen_p2 = self._world_to_screen(p2[0], p2[1])
                pygame.draw.line(self.screen, color, screen_p1, screen_p2, int(segment_width))

            # Draw circles at each point with appropriate width
            for i, point in enumerate(track.center_points):
                segment_width = track.get_road_width_at_segment(i)
                color = self.ROAD_COLOR if segment_width >= track.road_width else (80, 80, 50)
                screen_pos = self._world_to_screen(point[0], point[1])
                pygame.draw.circle(
                    self.screen,
                    color,
                    screen_pos,
                    int(segment_width // 2)
                )

    def _draw_checkpoints(self, track, next_checkpoint: int = None, visited_checkpoints: set = None):
        """Draw all checkpoints on the track."""
        if self.screen is None:
            return

        # Only draw if track has checkpoints attribute (F1TenthTrack)
        if not hasattr(track, 'checkpoints'):
            return

        visited = visited_checkpoints if visited_checkpoints is not None else set()

        for i, (cx, cy) in enumerate(track.checkpoints):
            screen_pos = self._world_to_screen(cx, cy)

            # Choose color based on checkpoint status
            if i == next_checkpoint:
                # Next checkpoint - GREEN (agent's target)
                color = self.NEXT_CHECKPOINT_COLOR
                radius = 8
            elif i in visited:
                # Visited checkpoint - GRAY
                color = self.VISITED_CHECKPOINT_COLOR
                radius = 4
            else:
                # Unvisited checkpoint - BLUE
                color = self.CHECKPOINT_COLOR
                radius = 4

            # Draw checkpoint circle
            pygame.draw.circle(self.screen, color, screen_pos, radius)

            # Draw border for next checkpoint
            if i == next_checkpoint:
                pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, radius + 2, 2)

    def _draw_finish_line(self, track: Track):
        """Draw the finish line."""
        if self.screen is None:
            return

        start, end = track.get_finish_line_points()
        screen_start = self._world_to_screen(start[0], start[1])
        screen_end = self._world_to_screen(end[0], end[1])
        pygame.draw.line(self.screen, self.FINISH_LINE_COLOR, screen_start, screen_end, 5)

    def _draw_obstacles(self, track: Track):
        """Draw obstacles (cones/barriers)."""
        if self.screen is None:
            return

        for obs_x, obs_y, obs_radius in track.obstacles:
            screen_pos = self._world_to_screen(obs_x, obs_y)
            # Draw obstacle as orange circle
            pygame.draw.circle(
                self.screen,
                self.OBSTACLE_COLOR,
                screen_pos,
                int(obs_radius)
            )
            # Add black outline
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                screen_pos,
                int(obs_radius),
                2  # thickness
            )

    def _draw_skid_marks(self, car: Car):
        """Draw skid marks left by drifting."""
        if self.screen is None or not car.skid_marks:
            return

        import math

        # Draw dual tire tracks (left and right wheels)
        for i, (x, y, angle) in enumerate(car.skid_marks):
            # Fade older marks
            alpha = int(255 * (i / len(car.skid_marks)))
            color = (20 + alpha // 10, 20 + alpha // 10, 20 + alpha // 10)

            # Calculate positions for left and right tire marks
            tire_offset = 8  # Distance from center
            left_x = x - math.cos(angle) * tire_offset
            left_y = y - math.sin(angle) * tire_offset
            right_x = x + math.cos(angle) * tire_offset
            right_y = y + math.sin(angle) * tire_offset

            # Apply camera offset
            screen_left = self._world_to_screen(left_x, left_y)
            screen_right = self._world_to_screen(right_x, right_y)

            # Draw small circles for tire marks
            pygame.draw.circle(self.screen, color, screen_left, 2)
            pygame.draw.circle(self.screen, color, screen_right, 2)

    def _draw_car(self, car: Car):
        """Draw the car as a rotated rectangle."""
        if self.screen is None:
            return

        corners = car.get_corners()
        # Apply camera offset to all corners
        screen_corners = [self._world_to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(self.screen, self.CAR_COLOR, screen_corners)

        front = (
            (corners[0][0] + corners[1][0]) / 2,
            (corners[0][1] + corners[1][1]) / 2
        )
        screen_front = self._world_to_screen(front[0], front[1])
        screen_center = self._world_to_screen(car.x, car.y)
        pygame.draw.line(self.screen, (255, 255, 255), screen_center, screen_front, 2)

    def _draw_ui(
        self,
        lap_time: float,
        lap_complete: bool,
        best_time: Optional[float],
        car: Car
    ):
        """Draw UI elements like timer and status."""
        if self.screen is None or self.font is None:
            return

        time_text = f"Time: {lap_time:.2f}s"
        time_surface = self.font.render(time_text, True, self.TEXT_COLOR)
        self.screen.blit(time_surface, (10, 10))

        if best_time is not None:
            best_text = f"Best: {best_time:.2f}s"
            best_surface = self.font.render(best_text, True, self.TEXT_COLOR)
            self.screen.blit(best_surface, (10, 50))

        speed_text = f"Speed: {abs(car.velocity):.0f}"
        speed_surface = self.font.render(speed_text, True, self.TEXT_COLOR)
        self.screen.blit(speed_surface, (10, 90))

        status = "ON ROAD" if car.on_road else "OFF ROAD"
        status_color = (0, 255, 0) if car.on_road else (255, 0, 0)
        status_surface = self.font.render(status, True, status_color)
        self.screen.blit(status_surface, (10, 130))

        # Show drift indicator
        if car.is_drifting:
            drift_text = "DRIFTING!"
            drift_color = (255, 165, 0)  # Orange
            drift_surface = self.font.render(drift_text, True, drift_color)
            self.screen.blit(drift_surface, (10, 170))

        if lap_complete:
            complete_text = f"LAP COMPLETE! Time: {lap_time:.2f}s"
            complete_surface = self.font.render(complete_text, True, self.FINISH_LINE_COLOR)
            text_rect = complete_surface.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(complete_surface, text_rect)

            restart_text = "Press R to restart"
            restart_surface = self.font.render(restart_text, True, self.TEXT_COLOR)
            restart_rect = restart_surface.get_rect(
                center=(self.width // 2, self.height // 2 + 40)
            )
            self.screen.blit(restart_surface, restart_rect)

    def _draw_debug_observations(self, car: Car, track: Track):
        """Draw debug visualization of what the agent observes (5-ray system)."""
        if self.screen is None or self.small_font is None:
            return

        import math

        # Get car state
        car_screen = self._world_to_screen(car.x, car.y)

        # Get ray data from environment (if available)
        # We need to access the environment's ray data
        # For now, we'll recalculate the rays here for visualization

        # Ray directions (same as in env.py)
        ray_angles_offset = [
            0.0,              # Front
            math.pi / 4,      # Front-right (45°)
            -math.pi / 4,     # Front-left (-45°)
            math.pi / 2,      # Right (90°)
            -math.pi / 2,     # Left (-90°)
        ]

        ray_colors = RAY_COLORS  # From config

        # Draw 5 distance rays and store normalized distances
        max_ray_length = RAY_MAX_DISTANCE  # From config
        ray_distances = []  # Store normalized distances for display

        for i, angle_offset in enumerate(ray_angles_offset):
            ray_angle = car.angle + angle_offset

            # Cast ray to find distance to track edge
            step_size = 5.0
            dx = math.sin(ray_angle) * step_size
            dy = -math.cos(ray_angle) * step_size

            current_x = car.x
            current_y = car.y
            distance = 0.0

            # Check if car is currently on road
            car_on_road = track.is_on_road(car.x, car.y)

            # Find where ray hits track edge (same logic as env.py)
            while distance < max_ray_length:
                current_x += dx
                current_y += dy
                distance += step_size

                current_on_road = track.is_on_road(current_x, current_y)

                # If car is ON road: detect when ray goes OFF road (track edge)
                if car_on_road and not current_on_road:
                    break

                # If car is OFF road: detect when ray goes ON road (distance to track)
                if not car_on_road and current_on_road:
                    break

            # Store normalized distance
            normalized_distance = min(distance / max_ray_length, 1.0)
            ray_distances.append(normalized_distance)

            # Draw the ray
            ray_end_screen = self._world_to_screen(current_x, current_y)
            pygame.draw.line(self.screen, ray_colors[i], car_screen, ray_end_screen, 2)

            # Draw endpoint circle
            pygame.draw.circle(self.screen, ray_colors[i], ray_end_screen, 4)

        # Draw text panel with 8 hybrid observations
        panel_x = self.width - 320
        panel_y = 10
        panel_width = 310
        panel_height = 220

        # Semi-transparent background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((0, 0, 0))
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # Calculate observations for display (simplified version)
        speed_norm = abs(car.velocity) / car.max_velocity

        # Display observations (matching env.py order)
        obs_labels = [
            "HYBRID OBSERVATIONS (8)",
            "--- Distance Sensors ---",
            f"1. Front ray: (normalized)",
            f"2. Front-right (45°): (normalized)",
            f"3. Front-left (-45°): (normalized)",
            f"4. Right (90°): (normalized)",
            f"5. Left (-90°): (normalized)",
            "--- Racing Line ---",
            f"6. Dist to centerline: (normalized)",
            f"7. Angle to centerline: (normalized)",
            f"8. Velocity: {speed_norm:.2f}",
        ]

        y_offset = panel_y + 10
        for i, label in enumerate(obs_labels):
            if i == 0:
                color = (255, 255, 0)  # Yellow title
            elif "---" in label:
                color = (100, 200, 255)  # Light blue for section headers
            else:
                color = (255, 255, 255)  # White for observations
            text_surface = self.small_font.render(label, True, color)
            self.screen.blit(text_surface, (panel_x + 5, y_offset))
            y_offset += 18

        # Legend at bottom
        legend_y = self.height - 120
        legend_items = [
            ("GREEN: Front ray", (0, 255, 0)),
            ("YELLOW: Front-right", (255, 255, 0)),
            ("CYAN: Front-left", (0, 255, 255)),
            ("ORANGE: Right ray", (255, 165, 0)),
            ("PURPLE: Left ray", (138, 43, 226)),
        ]

        y_offset = legend_y
        for text, color in legend_items:
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20

    def _draw_minimap(self, car: Car, track: Track):
        """Draw minimap showing entire track in corner."""
        if self.screen is None or self.small_font is None:
            return

        # Minimap settings
        minimap_size = 200
        minimap_x = self.width - minimap_size - 10
        minimap_y = self.height - minimap_size - 10

        # Semi-transparent background
        minimap_surface = pygame.Surface((minimap_size, minimap_size))
        minimap_surface.set_alpha(180)
        minimap_surface.fill((0, 0, 0))
        self.screen.blit(minimap_surface, (minimap_x, minimap_y))

        # Calculate scale to fit entire track
        track_width = getattr(track, 'width', 1000)
        track_height = getattr(track, 'height', 1000)
        scale = min(minimap_size / track_width, minimap_size / track_height) * 0.9

        # Draw centerline
        if hasattr(track, 'center_points') and track.center_points:
            for i in range(len(track.center_points)):
                p1 = track.center_points[i]
                p2 = track.center_points[(i + 1) % len(track.center_points)]

                # Scale to minimap coordinates
                x1 = minimap_x + int(p1[0] * scale)
                y1 = minimap_y + int(p1[1] * scale)
                x2 = minimap_x + int(p2[0] * scale)
                y2 = minimap_y + int(p2[1] * scale)

                pygame.draw.line(self.screen, (100, 100, 100), (x1, y1), (x2, y2), 2)

        # Draw car position
        car_x = minimap_x + int(car.x * scale)
        car_y = minimap_y + int(car.y * scale)
        pygame.draw.circle(self.screen, (255, 0, 0), (car_x, car_y), 3)
