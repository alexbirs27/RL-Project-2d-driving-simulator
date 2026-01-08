import pygame
from typing import List, Optional

from actions import Action
from car import Car, CarState
from track import Track
from f1tenth_track import F1TenthTrack
from renderer import Renderer


class GameEngine:
    """
    Main game engine that orchestrates all components.

    Separates physics updates, collision detection, and rendering.
    Exposes game state for RL integration.
    """

    def __init__(self, width: int = 1600, height: int = 1200, render: bool = True):
        self.width = width
        self.height = height

        # Use F1Tenth Spielberg track instead of procedural track
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(project_root, "tracks", "Spielberg", "Spielberg_centerline.csv")

        self.track = F1TenthTrack(
            csv_path=csv_path,
            scale=50.0,       # pixels per meter
            road_width=100.0  # track width in pixels
        )
        self.car = Car(
            self.track.start_position[0],
            self.track.start_position[1],
            self.track.start_angle
        )
        # Adjust car size for F1Tenth scale
        self.car.width = 20
        self.car.length = 14

        self.renderer = Renderer(width, height)
        self.renderer.enabled = render

        self.lap_time = 0.0
        self.lap_complete = False
        self.best_time: Optional[float] = None
        self.lap_started = False
        self.visited_checkpoints = set()  # Track visited checkpoints for valid lap completion
        self.hit_obstacle = False  # Track if obstacle was hit this step

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
        self.visited_checkpoints = set()
        self.visited_checkpoints.add(0)  # Start at checkpoint 0

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
            return self.car.get_state(self.lap_complete, self.lap_time, self.hit_obstacle)

        prev_x, prev_y = self.car.x, self.car.y

        self.car.update(dt, actions)

        # Reset hit_obstacle before checking
        self.hit_obstacle = False
        self._check_collisions()

        if self.lap_started:
            self.lap_time += dt

        if self.car.velocity > 10:
            self.lap_started = True

        # Track visited checkpoints
        current_checkpoint = self.track.get_nearest_checkpoint(self.car.x, self.car.y)
        self.visited_checkpoints.add(current_checkpoint)

        # Only count lap completion if:
        # 1. Lap has started
        # 2. Car crosses finish line in forward direction
        # 3. Car has visited at least 80% of checkpoints (prevents shortcut exploits)
        min_checkpoints_required = int(self.track.num_checkpoints * 0.8)
        if (self.lap_started and
            len(self.visited_checkpoints) >= min_checkpoints_required and
            self.track.check_finish_line(prev_x, prev_y, self.car.x, self.car.y)):
            self.lap_complete = True
            if self.best_time is None or self.lap_time < self.best_time:
                self.best_time = self.lap_time

        return self.car.get_state(self.lap_complete, self.lap_time, self.hit_obstacle)

    def _check_collisions(self):
        """Check and handle all collisions."""
        corners = self.car.get_corners()
        on_road = all(self.track.is_on_road(x, y) for x, y in corners)
        self.car.on_road = on_road

        # Check obstacle collision
        self.hit_obstacle = self.track.check_obstacle_collision(self.car.x, self.car.y, car_radius=15)

    def render(self):
        """Render the current game state."""
        # Set camera to follow car with zoom
        zoom_level = 1.5  # Adjust zoom for better track visibility
        self.renderer.camera_x = self.car.x - (self.renderer.width / zoom_level) // 2
        self.renderer.camera_y = self.car.y - (self.renderer.height / zoom_level) // 2
        self.renderer.zoom = zoom_level

        self.renderer.render(
            self.car,
            self.track,
            self.lap_time,
            self.lap_complete,
            self.best_time
        )

    def get_state(self) -> CarState:
        """Get the current game state."""
        return self.car.get_state(self.lap_complete, self.lap_time, self.hit_obstacle)

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
                elif event.key == pygame.K_d:
                    # Toggle debug visualization
                    self.renderer.show_debug = not self.renderer.show_debug
                    print(f"Debug view: {'ON' if self.renderer.show_debug else 'OFF'}")

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
