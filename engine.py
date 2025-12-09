import pygame
from typing import List, Optional

from actions import Action
from car import Car, CarState
from track import Track
from renderer import Renderer


class GameEngine:
    """
    Main game engine that orchestrates all components.

    Separates physics updates, collision detection, and rendering.
    Exposes game state for RL integration.
    """

    def __init__(self, width: int = 1200, height: int = 800, render: bool = True):
        self.width = width
        self.height = height

        self.track = Track(width, height)
        self.car = Car(
            self.track.start_position[0],
            self.track.start_position[1],
            self.track.start_angle
        )

        self.renderer = Renderer(width, height)
        self.renderer.enabled = render

        self.lap_time = 0.0
        self.lap_complete = False
        self.best_time: Optional[float] = None
        self.lap_started = False

        self.running = True

    def init(self):
        """Initialize pygame and all subsystems."""
        pygame.init()
        self.renderer.init()
        self.reset()

    def reset(self):
        """Reset the game state for a new lap."""
        self.car.reset(
            self.track.start_position[0],
            self.track.start_position[1],
            self.track.start_angle
        )
        self.lap_time = 0.0
        self.lap_complete = False
        self.lap_started = False

    def step(self, actions: List[Action], dt: float) -> CarState:
        """
        Perform one simulation step.

        Args:
            actions: List of actions to apply.
            dt: Delta time in seconds.

        Returns:
            Current car state after the step.
        """
        if self.lap_complete:
            return self.car.get_state(self.lap_complete, self.lap_time)

        prev_x, prev_y = self.car.x, self.car.y

        self.car.update(dt, actions)

        self._check_collisions()

        if self.lap_started:
            self.lap_time += dt

        if self.car.velocity > 10:
            self.lap_started = True

        if self.lap_started and self.track.check_finish_line(
            prev_x, prev_y, self.car.x, self.car.y
        ):
            self.lap_complete = True
            if self.best_time is None or self.lap_time < self.best_time:
                self.best_time = self.lap_time

        return self.car.get_state(self.lap_complete, self.lap_time)

    def _check_collisions(self):
        """Check and handle all collisions."""
        corners = self.car.get_corners()
        on_road = all(self.track.is_on_road(x, y) for x, y in corners)
        self.car.on_road = on_road

    def render(self):
        """Render the current game state."""
        self.renderer.render(
            self.car,
            self.track,
            self.lap_time,
            self.lap_complete,
            self.best_time
        )

    def get_state(self) -> CarState:
        """Get the current game state."""
        return self.car.get_state(self.lap_complete, self.lap_time)

    def handle_events(self) -> List[Action]:
        """
        Handle pygame events and return actions from keyboard input.

        Returns:
            List of actions based on current keyboard state.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()

        actions = []
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            actions.append(Action.ACCELERATE)
        if keys[pygame.K_DOWN]:
            actions.append(Action.BRAKE)
        if keys[pygame.K_LEFT]:
            actions.append(Action.TURN_LEFT)
        if keys[pygame.K_RIGHT]:
            actions.append(Action.TURN_RIGHT)

        return actions

    def quit(self):
        """Clean up and quit pygame."""
        pygame.quit()
