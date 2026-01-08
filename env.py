"""
Gymnasium-compatible environment wrapper for the racing game.

This environment can be used by ALL RL algorithms (DQN, PPO, A2C, etc.)

Observation Space (10 values - SIMPLIFIED and FOCUSED):
    Distance Sensors (5 values) - LONGER RAYS (400px):
        1. Front ray distance (normalized 0-1)
        2. Front-right ray (45°, normalized 0-1)
        3. Front-left ray (-45°, normalized 0-1)
        4. Right ray (90°, normalized 0-1)
        5. Left ray (-90°, normalized 0-1)

    Navigation to Next Checkpoint (2 values) - WHERE TO GO:
        6. Distance to next checkpoint (normalized 0-1, max 600px)
        7. Angle to next checkpoint (normalized -1 to 1) - tells agent which direction to drive

    Racing Line Awareness (3 values):
        8. Distance to centerline (normalized 0-1): 0=perfect line, 1=track edge
        9. Angle to centerline (normalized -1 to 1): heading alignment with track
        10. Velocity (normalized 0-1): current speed

Action Space (9 discrete actions):
    0: Nothing
    1: Accelerate
    2: Brake
    3: Turn left
    4: Turn right
    5: Accelerate + Turn left
    6: Accelerate + Turn right
    7: Brake + Turn left
    8: Brake + Turn right

Reward Structure (SIMPLIFIED - 5 core components):
    - +10.0 per new checkpoint visited (forward progress only)
    - +0.5 for speed (velocity-based, encourages going fast)
    - +0.3 for staying near centerline (racing line bonus)
    - -1.0 per step off-road (teaches to stay on track)
    - +100.0 for completing lap (goal achievement)

Previous issues FIXED:
    - Removed 15+ conflicting reward components
    - Removed anticipatory braking reward (was confusing)
    - Removed drift/corner rewards (too complex)
    - Removed time-based checkpoint rewards (caused exploitation)
    - Checkpoint reward only given for FORWARD progress (prevents reverse exploit)

Early Termination:
    - Episode ends if stuck off-road for 10 seconds (600 steps) - allows recovery learning
    - Episode ends if no progress for 500 steps
    - Episode ends after max_steps (default 10000)

Usage:
    from env import make_env
    env = make_env(render_mode="human")  # or None for headless
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from game import GameEngine, Action
from config import (
    OBS_DIM, RAY_MAX_DISTANCE, CHECKPOINT_NAV_MAX_DISTANCE,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_LIMIT, OFFROAD_TRUNCATION_LIMIT,
    REWARD_CHECKPOINT, REWARD_SPEED, REWARD_CENTERLINE, REWARD_OFFROAD, REWARD_LAP_COMPLETE
)

#    Gymnasium environment wrapper for the 2D racing game.

class RacingEnv(gym.Env):

    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS_PER_EPISODE

        self.engine = GameEngine(
            render=(render_mode == "human")
        )

        # 9 actions: combinations of acceleration/brake with steering
        self.action_space = spaces.Discrete(9)

        # Observations from config
        # [5 ray distances, next_checkpoint_distance, angle_to_next_checkpoint,
        #  dist_to_centerline, angle_to_centerline, velocity]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # For visualization
        self.last_ray_angles = []
        self.last_ray_distances = []

        # Action mapping: allows combined actions
        self.action_map = {
            0: [],                                      # Nothing
            1: [Action.ACCELERATE],                     # Accelerate
            2: [Action.BRAKE],                          # Brake
            3: [Action.TURN_LEFT],                      # Turn left
            4: [Action.TURN_RIGHT],                     # Turn right
            5: [Action.ACCELERATE, Action.TURN_LEFT],   # Accelerate + Left
            6: [Action.ACCELERATE, Action.TURN_RIGHT],  # Accelerate + Right
            7: [Action.BRAKE, Action.TURN_LEFT],        # Brake + Left
            8: [Action.BRAKE, Action.TURN_RIGHT],       # Brake + Right
        }

        self.dt = 1.0 / 60.0
        self.current_step = 0
        self.last_checkpoint = 0
        self.visited_checkpoints = set()
        self.no_progress_steps = 0
        self.last_steering_action = 0  # Track steering for smoothness reward
        self.checkpoint_times = {}  # Track time when each checkpoint was reached
        self.current_time = 0.0  # Track elapsed time in episode
        self.consecutive_offroad_steps = 0  # Track how long stuck off-road

        # Tracking for observation space (acceleration and turning rate)
        self.last_velocity = 0.0  # For acceleration calculation
        self.last_angle = 0.0  # For turning rate calculation

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Reset the environment.
        super().reset(seed=seed)

        if not hasattr(self, '_initialized'):
            self.engine.init()
            self._initialized = True
        else:
            self.engine.reset()

        self.current_step = 0
        self.last_checkpoint = 0
        self.next_checkpoint = 1  # Track the NEXT checkpoint we need to reach
        self.visited_checkpoints = set()
        self.visited_checkpoints.add(0)
        self.no_progress_steps = 0
        self.last_steering_action = 0  # Reset steering tracker
        self.current_time = 0.0  # Reset time
        self.consecutive_offroad_steps = 0  # Reset off-road counter

        # Reset tracking for observations
        self.last_velocity = self.engine.car.velocity
        self.last_angle = self.engine.car.angle

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Execute one environment step.
        actions = self.action_map[action]
        state = self.engine.step(actions, self.dt)
        self.current_step += 1
        self.current_time += self.dt  # Track elapsed time

        # Track consecutive off-road steps
        if not state.on_road:
            self.consecutive_offroad_steps += 1
        else:
            self.consecutive_offroad_steps = 0  # Reset counter when back on road

        obs = self._get_observation()
        reward = self._calculate_reward(state, action)
        terminated = state.lap_complete
        truncated = self._check_truncation(state)
        info = self._get_info()

        self.last_angle = state.angle

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _cast_ray(self, x: float, y: float, angle: float, max_distance: float = None) -> float:
        """
        Cast a ray from (x, y) in direction 'angle' and return distance to track edge.

        Args:
            x, y: Starting position
            angle: Direction to cast ray (in radians)
            max_distance: Maximum ray length (from config)

        Returns:
            Distance to track edge (normalized 0-1, where 1 = max_distance)
        """
        if max_distance is None:
            max_distance = RAY_MAX_DISTANCE

        track = self.engine.track
        step_size = 5.0  # Check every 5 pixels

        dx = math.sin(angle) * step_size
        dy = -math.cos(angle) * step_size

        current_x = x
        current_y = y
        distance = 0.0

        # Check if car is currently on road
        car_on_road = track.is_on_road(x, y)

        # Cast ray until we hit track edge or max distance
        while distance < max_distance:
            current_x += dx
            current_y += dy
            distance += step_size

            current_on_road = track.is_on_road(current_x, current_y)

            # If car is ON road: detect when ray goes OFF road (track edge)
            if car_on_road and not current_on_road:
                return distance / max_distance  # Normalize to [0, 1]

            # If car is OFF road: detect when ray goes ON road (distance to track)
            if not car_on_road and current_on_road:
                return distance / max_distance  # Normalize to [0, 1]

        return 1.0  # Hit max distance, return normalized max

    def _get_distance_to_centerline(self, x: float, y: float) -> tuple[float, int]:
        """
        Calculate distance from position to nearest centerline point.

        Returns:
            (distance, nearest_index): Distance in pixels and index of nearest centerline point
        """
        track = self.engine.track
        min_dist = float('inf')
        nearest_idx = 0

        for i, (cx, cy) in enumerate(track.center_points):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return min_dist, nearest_idx

    def _get_angle_to_centerline(self, x: float, y: float, car_angle: float, nearest_idx: int) -> float:
        """
        Calculate angle difference between car heading and centerline direction.

        Args:
            x, y: Car position
            car_angle: Car heading in radians
            nearest_idx: Index of nearest centerline point

        Returns:
            Angle difference in radians (-π to π)
        """
        track = self.engine.track

        # Get tangent direction of centerline at nearest point
        # Use next point to calculate direction
        next_idx = (nearest_idx + 1) % len(track.center_points)
        cx1, cy1 = track.center_points[nearest_idx]
        cx2, cy2 = track.center_points[next_idx]

        # Centerline direction vector
        dx = cx2 - cx1
        dy = cy2 - cy1
        centerline_angle = math.atan2(dx, -dy)  # Same convention as car angle

        # Calculate angle difference (normalize to -π to π)
        angle_diff = car_angle - centerline_angle
        # Normalize to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

    def _get_lookahead_corners(self, car_x: float, car_y: float, nearest_idx: int) -> tuple:
        """
        Get lookahead information for next 3 corners.

        Returns:
            (d1, d2, d3, c1, c2, c3): Distances and curvatures for 3 lookahead points
        """
        track = self.engine.track

        # Sample 3 points ahead on centerline
        # Spacing: sample at 30, 60, 90 checkpoints ahead (adjustable based on track)
        lookahead_offsets = [30, 60, 90]

        distances = []
        curvatures = []

        for offset in lookahead_offsets:
            # Get lookahead point index
            lookahead_idx = (nearest_idx + offset) % len(track.center_points)

            # Calculate distance from car to lookahead point
            lx, ly = track.center_points[lookahead_idx]
            dist = math.sqrt((car_x - lx) ** 2 + (car_y - ly) ** 2)
            distances.append(dist)

            # Calculate curvature at lookahead point
            # Curvature = angle change from previous to next segment
            prev_idx = (lookahead_idx - 1) % len(track.center_points)
            next_idx = (lookahead_idx + 1) % len(track.center_points)

            px, py = track.center_points[prev_idx]
            cx, cy = track.center_points[lookahead_idx]
            nx, ny = track.center_points[next_idx]

            # Angle of incoming segment
            angle_in = math.atan2(cx - px, -(cy - py))
            # Angle of outgoing segment
            angle_out = math.atan2(nx - cx, -(ny - cy))

            # Curvature is the angle change
            curvature = angle_out - angle_in
            # Normalize to [-π, π]
            while curvature > math.pi:
                curvature -= 2 * math.pi
            while curvature < -math.pi:
                curvature += 2 * math.pi

            curvatures.append(curvature)

        return (*distances, *curvatures)

    def _get_observation(self) -> np.ndarray:
        """
        Improved observation with navigation guidance.

        Returns 10 values:
            [front_ray, front_right_ray, front_left_ray, right_ray, left_ray,
             next_checkpoint_distance, angle_to_next_checkpoint,
             dist_to_centerline, angle_to_centerline, velocity]
        """
        state = self.engine.get_state()
        car = self.engine.car
        track = self.engine.track

        # === 1-5. Cast 5 rays for collision avoidance (400px range) ===
        ray_angles = [
            0.0,              # Front
            math.pi / 4,      # Front-right (45°)
            -math.pi / 4,     # Front-left (-45°)
            math.pi / 2,      # Right (90°)
            -math.pi / 2,     # Left (-90°)
        ]

        ray_distances = []
        for angle_offset in ray_angles:
            ray_angle = state.angle + angle_offset
            distance = self._cast_ray(state.x, state.y, ray_angle)
            ray_distances.append(distance)

        # === 6-7. Navigation to NEXT checkpoint (tells agent WHERE TO GO) ===
        # Get position of next checkpoint
        next_cp_pos = track.get_checkpoint_position(self.next_checkpoint)

        # Distance to next checkpoint
        dx = next_cp_pos[0] - state.x
        dy = next_cp_pos[1] - state.y
        dist_to_next_cp = math.sqrt(dx * dx + dy * dy)
        dist_to_next_cp_norm = np.clip(dist_to_next_cp / CHECKPOINT_NAV_MAX_DISTANCE, 0, 1)

        # Angle to next checkpoint (tells agent which direction to turn)
        angle_to_next_cp = math.atan2(dx, -dy)  # Angle of checkpoint in world
        angle_diff = angle_to_next_cp - state.angle  # Difference from car's heading
        # Normalize to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        angle_to_next_cp_norm = np.clip(angle_diff / math.pi, -1, 1)

        # === 8. Distance to centerline (normalized 0-1) ===
        dist_to_center, nearest_idx = self._get_distance_to_centerline(state.x, state.y)
        track_half_width = track.road_width / 2
        dist_to_center_norm = np.clip(dist_to_center / track_half_width, 0, 1)

        # === 9. Angle to centerline (normalized -1 to 1) ===
        angle_to_center = self._get_angle_to_centerline(state.x, state.y, state.angle, nearest_idx)
        angle_to_center_norm = np.clip(angle_to_center / math.pi, -1, 1)

        # === 10. Velocity (normalized 0-1) ===
        velocity_norm = np.clip(abs(state.velocity) / car.max_velocity, 0, 1)

        # Construct observation vector
        obs = np.array([
            ray_distances[0],         # 1. Front ray
            ray_distances[1],         # 2. Front-right ray
            ray_distances[2],         # 3. Front-left ray
            ray_distances[3],         # 4. Right ray
            ray_distances[4],         # 5. Left ray
            dist_to_next_cp_norm,     # 6. Distance to next checkpoint
            angle_to_next_cp_norm,    # 7. Angle to next checkpoint
            dist_to_center_norm,      # 8. Distance to centerline
            angle_to_center_norm,     # 9. Angle to centerline
            velocity_norm,            # 10. Velocity
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, state, action: int) -> float:
        """
        SIMPLIFIED reward function - 5 core components only.

        Design philosophy:
        1. Make forward progress through checkpoints (main goal)
        2. Go fast (racing is about speed)
        3. Stay on racing line (optimal path)
        4. Avoid going off-road (safety)
        5. Complete the lap (ultimate goal)
        """
        reward = 0.0
        track = self.engine.track
        car = self.engine.car

        current_cp = track.get_nearest_checkpoint(state.x, state.y)

        # ========================================
        # 1. CHECKPOINT PROGRESS (Primary Goal)
        # ========================================
        # Only reward if we reached the NEXT checkpoint (prevents reverse exploit)
        if current_cp == self.next_checkpoint:
            reward += REWARD_CHECKPOINT
            self.visited_checkpoints.add(current_cp)
            self.next_checkpoint = (current_cp + 1) % track.num_checkpoints
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        self.last_checkpoint = current_cp

        # ========================================
        # 2. SPEED REWARD (Go Fast!)
        # ========================================
        # Simple velocity reward (encourages speed)
        if state.on_road:
            speed_reward = (abs(state.velocity) / car.max_velocity) * REWARD_SPEED
            reward += speed_reward

        # ========================================
        # 3. RACING LINE BONUS
        # ========================================
        # Reward staying near centerline
        if state.on_road:
            dist_to_center, _ = self._get_distance_to_centerline(state.x, state.y)
            track_half_width = track.road_width / 2
            # Closer to center = better (0 = perfect, 1 = edge)
            centerline_score = 1.0 - (dist_to_center / track_half_width)
            reward += centerline_score * REWARD_CENTERLINE

        # ========================================
        # 4. OFF-ROAD PENALTY
        # ========================================
        if not state.on_road:
            reward += REWARD_OFFROAD  # Note: REWARD_OFFROAD is negative in config

        # ========================================
        # 5. LAP COMPLETION BONUS
        # ========================================
        if state.lap_complete:
            reward += REWARD_LAP_COMPLETE

        return reward

    def _check_truncation(self, state) -> bool:
        #Check if episode should be truncated
        if self.current_step >= self.max_steps:
            return True

        # Truncate if no progress for too long
        if self.no_progress_steps > NO_PROGRESS_LIMIT:
            return True

        # Truncate if stuck off-road for too long
        if self.consecutive_offroad_steps > OFFROAD_TRUNCATION_LIMIT:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        state = self.engine.get_state()
        return {
            "lap_time": state.lap_time,
            "lap_complete": state.lap_complete,
            "on_road": state.on_road,
            "velocity": state.velocity,
            "checkpoint": self.last_checkpoint,
            "checkpoints_visited": len(self.visited_checkpoints),
            "step": self.current_step
        }

    def render(self):
        #Render the environment
        if self.render_mode == "human":
            self.engine.render(
                next_checkpoint=self.next_checkpoint,
                visited_checkpoints=self.visited_checkpoints
            )
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

    def close(self):
        self.engine.quit()


def make_env(render_mode: Optional[str] = None, max_steps: int = None):
    """
    Factory function to create the environment.

    Args:
        render_mode: "human" for visual rendering, None for headless
        max_steps: Maximum steps per episode

    Returns:
        RacingEnv instance
    """
    return RacingEnv(render_mode=render_mode, max_steps=max_steps)
