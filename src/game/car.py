import math
from dataclasses import dataclass
from typing import List, Tuple

from src.game.actions import Action


@dataclass
class CarState:
    """Represents the complete state of the car."""
    x: float
    y: float
    angle: float
    velocity: float
    on_road: bool
    lap_complete: bool
    lap_time: float


class Car:
    """
    Car with basic physics simulation.

    Handles position, velocity, angle, acceleration, and friction.
    """

    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0

        self.width = 20
        self.height = 40

        self.max_velocity = 240.0
        self.acceleration = 150.0
        self.brake_force = 200.0
        self.friction = 30.0
        self.turn_speed = 3.0

        self.offroad_friction_multiplier = 2.5
        self.offroad_max_velocity = 60.0

        self.on_road = True

    def reset(self, x: float, y: float, angle: float = 0.0):
        """Reset car to starting position."""
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.on_road = True

    def update(self, dt: float, actions: List[Action]):
        """
        Update car physics based on actions and delta time.

        Args:
            dt: Delta time in seconds.
            actions: List of actions to apply this frame.
        """
        for action in actions:
            if action == Action.ACCELERATE:
                self.velocity += self.acceleration * dt
            elif action == Action.BRAKE:
                self.velocity -= self.brake_force * dt
            elif action == Action.TURN_LEFT:
                if abs(self.velocity) > 1.0:
                    self.angle -= self.turn_speed * dt * (self.velocity / self.max_velocity)
            elif action == Action.TURN_RIGHT:
                if abs(self.velocity) > 1.0:
                    self.angle += self.turn_speed * dt * (self.velocity / self.max_velocity)

        current_friction = self.friction
        current_max_velocity = self.max_velocity

        if not self.on_road:
            current_friction *= self.offroad_friction_multiplier
            current_max_velocity = self.offroad_max_velocity

        if self.velocity > 0:
            self.velocity -= current_friction * dt
            self.velocity = max(0, self.velocity)
        elif self.velocity < 0:
            self.velocity += current_friction * dt
            self.velocity = min(0, self.velocity)

        self.velocity = max(-current_max_velocity * 0.3,
                           min(current_max_velocity, self.velocity))

        self.x += math.sin(self.angle) * self.velocity * dt
        self.y -= math.cos(self.angle) * self.velocity * dt

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get the four corners of the car for collision detection."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        hw = self.width / 2
        hh = self.height / 2

        corners = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh)
        ]

        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated.append((rx, ry))

        return rotated

    def get_state(self, lap_complete: bool, lap_time: float) -> CarState:
        """Get the current state of the car."""
        return CarState(
            x=self.x,
            y=self.y,
            angle=self.angle,
            velocity=self.velocity,
            on_road=self.on_road,
            lap_complete=lap_complete,
            lap_time=lap_time
        )
