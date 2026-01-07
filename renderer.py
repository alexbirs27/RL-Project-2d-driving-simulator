import pygame
from typing import Optional

from car import Car
from track import Track


class Renderer:
    """
    Handles all rendering operations.

    Can be disabled entirely for headless simulation during RL training.
    """

    GRASS_COLOR = (34, 139, 34)
    ROAD_COLOR = (60, 60, 60)
    ROAD_EDGE_COLOR = (255, 255, 255)
    CAR_COLOR = (220, 20, 60)
    FINISH_LINE_COLOR = (255, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    SKID_MARK_COLOR = (20, 20, 20)  # Dark tire marks
    OBSTACLE_COLOR = (255, 100, 0)  # Orange cones/barriers
    NARROW_SECTION_COLOR = (255, 200, 0)  # Yellow marking for narrow sections

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.enabled = True
        self.camera_x = 0
        self.camera_y = 0

    def init(self):
        """Initialize pygame display."""
        if not self.enabled:
            return

        pygame.display.set_caption("2D Racing Engine")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)

    def _world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates with camera offset."""
        return (int(x - self.camera_x), int(y - self.camera_y))

    def render(
        self,
        car: Car,
        track: Track,
        lap_time: float,
        lap_complete: bool,
        best_time: Optional[float]
    ):
        """Render the complete game state."""
        if not self.enabled or self.screen is None:
            return

        # Calculate camera offset to center on car
        self.camera_x = car.x - self.width // 2
        self.camera_y = car.y - self.height // 2

        self.screen.fill(self.GRASS_COLOR)
        self._draw_track(track)
        self._draw_obstacles(track)  # Draw obstacles
        self._draw_finish_line(track)
        self._draw_skid_marks(car)  # Draw skid marks before car
        self._draw_car(car)
        self._draw_ui(lap_time, lap_complete, best_time, car)

        pygame.display.flip()

    def _draw_track(self, track: Track):
        """Draw the road surface with variable width sections."""
        if self.screen is None:
            return

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
