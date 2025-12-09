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

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.enabled = True

    def init(self):
        """Initialize pygame display."""
        if not self.enabled:
            return

        pygame.display.set_caption("2D Racing Engine")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)

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

        self.screen.fill(self.GRASS_COLOR)
        self._draw_track(track)
        self._draw_finish_line(track)
        self._draw_car(car)
        self._draw_ui(lap_time, lap_complete, best_time, car)

        pygame.display.flip()

    def _draw_track(self, track: Track):
        """Draw the road surface."""
        if self.screen is None:
            return

        for i in range(len(track.center_points)):
            p1 = track.center_points[i]
            p2 = track.center_points[(i + 1) % len(track.center_points)]
            pygame.draw.line(self.screen, self.ROAD_COLOR, p1, p2, track.road_width)

        for point in track.center_points:
            pygame.draw.circle(
                self.screen,
                self.ROAD_COLOR,
                (int(point[0]), int(point[1])),
                track.road_width // 2
            )

    def _draw_finish_line(self, track: Track):
        """Draw the finish line."""
        if self.screen is None:
            return

        start, end = track.get_finish_line_points()
        pygame.draw.line(self.screen, self.FINISH_LINE_COLOR, start, end, 5)

    def _draw_car(self, car: Car):
        """Draw the car as a rotated rectangle."""
        if self.screen is None:
            return

        corners = car.get_corners()
        pygame.draw.polygon(self.screen, self.CAR_COLOR, corners)

        front = (
            (corners[0][0] + corners[1][0]) / 2,
            (corners[0][1] + corners[1][1]) / 2
        )
        center = (car.x, car.y)
        pygame.draw.line(self.screen, (255, 255, 255), center, front, 2)

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
