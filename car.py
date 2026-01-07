import math
from dataclasses import dataclass
from typing import List, Tuple

from actions import Action


@dataclass
class CarState:
    """Represents the complete state of the car."""
    x: float
    y: float
    angle: float
    velocity: float
    lateral_velocity: float  # Drift velocity
    on_road: bool
    hit_obstacle: bool  # NEW: Did car hit obstacle this step?
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
        self.lateral_velocity = 0.0  # NEW: Sideways drift velocity

        self.width = 20
        self.height = 40

        self.max_velocity = 240.0
        self.acceleration = 150.0
        self.brake_force = 200.0
        self.friction = 30.0
        self.turn_speed = 3.0

        # RWD drift physics!
        self.grip_threshold = 150.0   # Speed at which rear loses grip when turning hard (higher = drift at higher speeds only)
        self.lateral_friction = 120.0  # How quickly drift is reduced (higher = less sliding)
        self.drift_strength = 5.0     # How much the rear kicks out (lower = less drift)

        self.offroad_friction_multiplier = 2.5
        self.offroad_max_velocity = 60.0

        self.on_road = True

        # Skid marks tracking
        self.skid_marks = []  # List of (x, y) positions where car is drifting
        self.is_drifting = False

    def reset(self, x: float, y: float, angle: float = 0.0):
        """Reset car to starting position."""
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.lateral_velocity = 0.0
        self.on_road = True
        self.skid_marks = []
        self.is_drifting = False

    def update(self, dt: float, actions: List[Action]):
        """
        Update car physics based on actions and delta time.

        Args:
            dt: Delta time in seconds.
            actions: List of actions to apply this frame.
        """
        # Track angle change for drift calculation
        angle_before = self.angle

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

        # Realistic RWD drift: rear slides out when turning too hard at high speed
        angle_change = self.angle - angle_before

        # Only drift on road, not on grass
        if self.on_road and abs(self.velocity) > self.grip_threshold and abs(angle_change) > 0.001:
            # Calculate oversteer: how much the rear is sliding
            # Sharp turns at high speed = more oversteer
            turn_sharpness = abs(angle_change) / dt if dt > 0 else 0
            speed_over_grip = (abs(self.velocity) - self.grip_threshold) / self.max_velocity

            # Rear slides out proportional to turn sharpness and speed
            oversteer_amount = turn_sharpness * speed_over_grip * self.drift_strength

            # Apply lateral velocity in direction of turn (rear kicks out)
            if angle_change > 0:  # Turning right
                self.lateral_velocity -= oversteer_amount
            else:  # Turning left
                self.lateral_velocity += oversteer_amount

        # Track if we're currently drifting (for visual effects)
        self.is_drifting = abs(self.lateral_velocity) > 5.0 and abs(self.velocity) > 80.0

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

        # Apply lateral friction to reduce drift
        # More friction off-road (harder to slide on grass)
        effective_lateral_friction = self.lateral_friction
        if not self.on_road:
            effective_lateral_friction *= 2.0  # Grass grips more (no smooth drift)

        if self.lateral_velocity > 0:
            self.lateral_velocity -= effective_lateral_friction * dt
            self.lateral_velocity = max(0, self.lateral_velocity)
        elif self.lateral_velocity < 0:
            self.lateral_velocity += effective_lateral_friction * dt
            self.lateral_velocity = min(0, self.lateral_velocity)

        # Update position with both forward velocity AND lateral drift
        # Forward direction
        forward_dx = math.sin(self.angle) * self.velocity * dt
        forward_dy = -math.cos(self.angle) * self.velocity * dt

        # Lateral direction (perpendicular to forward)
        lateral_dx = math.cos(self.angle) * self.lateral_velocity * dt
        lateral_dy = math.sin(self.angle) * self.lateral_velocity * dt

        # Combine both
        self.x += forward_dx + lateral_dx
        self.y += forward_dy + lateral_dy

        # Add skid marks when drifting
        if self.is_drifting and self.on_road:
            # Add position to skid marks (limit to last 500 points to avoid memory issues)
            self.skid_marks.append((self.x, self.y, self.angle))
            if len(self.skid_marks) > 500:
                self.skid_marks.pop(0)

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

    def get_state(self, lap_complete: bool, lap_time: float, hit_obstacle: bool = False) -> CarState:
        """Get the current state of the car."""
        return CarState(
            x=self.x,
            y=self.y,
            angle=self.angle,
            velocity=self.velocity,
            lateral_velocity=self.lateral_velocity,
            on_road=self.on_road,
            hit_obstacle=hit_obstacle,
            lap_complete=lap_complete,
            lap_time=lap_time
        )
