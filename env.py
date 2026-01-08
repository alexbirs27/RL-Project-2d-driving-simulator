"""
Gymnasium-compatible environment wrapper for the racing game.

This environment can be used by ALL RL algorithms (DQN, PPO, A2C, etc.)

Observation Space (8 values - Hybrid ray + racing line approach):
    Distance Sensors (5 values):
        1. Front ray distance (normalized 0-1)
        2. Front-right ray (45°, normalized 0-1)
        3. Front-left ray (-45°, normalized 0-1)
        4. Right ray (90°, normalized 0-1)
        5. Left ray (-90°, normalized 0-1)

    Racing Line Awareness (3 values):
        6. Distance to centerline (normalized 0-1): 0=perfect line, 1=track edge
        7. Angle to centerline (normalized -1 to 1): heading alignment with track
        8. Velocity (normalized 0-1): current speed

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

Reward Structure:
    - +20.0 per new checkpoint visited (forward progress)
    - +up to 5.0 for reaching checkpoint quickly (speed bonus)
    - +1.0 for speed (when on road)
    - +0.2 for staying on road
    - -0.5 for going off road
    - -0.1 for standing still
    - +0.1 for accelerating (encourages movement)
    - +0.5 for corner speed (maintaining speed through turns)
    - -0.15 for steering zigzag (encourages smooth racing lines)
    - +0.3 for controlled drift at high speed (advanced technique)
    - -50.0 for hitting obstacles (HEAVY PENALTY!)
    - -100.0 for getting stuck off-road (5+ seconds off-road = early termination)
    - +500 + time_bonus for completing lap (faster = better)

Early Termination:
    - Episode ends if stuck off-road for 5 seconds (300 steps)
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

#    Gymnasium environment wrapper for the 2D racing game.

class RacingEnv(gym.Env):

    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 3000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.engine = GameEngine(
            width=1600,
            height=1200,
            render=(render_mode == "human")
        )

        # 9 actions: combinations of acceleration/brake with steering
        self.action_space = spaces.Discrete(9)

        # 8 observations: Hybrid approach (rays + racing line)
        # [5 ray distances, dist_to_centerline, angle_to_centerline, velocity]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
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
        self.visited_checkpoints = set()
        self.visited_checkpoints.add(0)
        self.no_progress_steps = 0
        self._last_dist_to_next = None  # Reset distance tracker
        self.last_steering_action = 0  # Reset steering tracker
        self.checkpoint_times = {0: 0.0}  # Start at checkpoint 0 at time 0
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

    def _cast_ray(self, x: float, y: float, angle: float, max_distance: float = 200.0) -> float:
        """
        Cast a ray from (x, y) in direction 'angle' and return distance to track edge.

        Args:
            x, y: Starting position
            angle: Direction to cast ray (in radians)
            max_distance: Maximum ray length

        Returns:
            Distance to track edge (normalized 0-1, where 1 = max_distance)
        """
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
        Hybrid observation: rays for collision + centerline for racing line.

        Returns 8 values:
            [front_ray, front_right_ray, front_left_ray, right_ray, left_ray,
             dist_to_centerline, angle_to_centerline, velocity]
        """
        state = self.engine.get_state()
        car = self.engine.car

        # === 1-5. Cast 5 rays for collision avoidance ===
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
            distance = self._cast_ray(state.x, state.y, ray_angle, max_distance=200.0)
            ray_distances.append(distance)

        # === 6. Distance to centerline (normalized 0-1) ===
        dist_to_center, nearest_idx = self._get_distance_to_centerline(state.x, state.y)
        track_half_width = self.engine.track.road_width / 2
        dist_to_center_norm = np.clip(dist_to_center / track_half_width, 0, 1)

        # === 7. Angle to centerline (normalized -1 to 1) ===
        angle_to_center = self._get_angle_to_centerline(state.x, state.y, state.angle, nearest_idx)
        angle_to_center_norm = np.clip(angle_to_center / math.pi, -1, 1)

        # === 8. Velocity (normalized 0-1) ===
        velocity_norm = np.clip(abs(state.velocity) / car.max_velocity, 0, 1)

        # Construct observation vector
        obs = np.array([
            ray_distances[0],        # 1. Front ray
            ray_distances[1],        # 2. Front-right ray
            ray_distances[2],        # 3. Front-left ray
            ray_distances[3],        # 4. Right ray
            ray_distances[4],        # 5. Left ray
            dist_to_center_norm,     # 6. Distance to centerline
            angle_to_center_norm,    # 7. Angle to centerline
            velocity_norm,           # 8. Velocity
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, state, action: int) -> float:
        """
        Improved reward function with learning signals for braking and recovery.

        Core principles:
        1. Make progress (checkpoints)
        2. Go fast (speed) - but only when safe
        3. Brake before corners (anticipatory reward based on front ray)
        4. Stay on racing line (centerline distance)
        5. Don't crash (speed-dependent off-road penalty)
        6. Recover when off-road (gradient toward road)
        """
        reward = 0.0
        track = self.engine.track
        car = self.engine.car

        current_cp = track.get_nearest_checkpoint(state.x, state.y)

        # ========================================
        # 1. CHECKPOINT PROGRESS (Primary Goal)
        # ========================================
        if current_cp not in self.visited_checkpoints:
            diff = (current_cp - self.last_checkpoint) % track.num_checkpoints

            if diff > 0 and diff < track.num_checkpoints // 2:
                # Main reward: forward progress
                reward += 10.0 * diff
                self.visited_checkpoints.add(current_cp)
                self.no_progress_steps = 0
            elif diff > track.num_checkpoints // 2:
                # Backward movement penalty
                reward -= 5.0
                self.no_progress_steps += 1
        else:
            self.no_progress_steps += 1

        self.last_checkpoint = current_cp

        # ========================================
        # 2. SPEED (Go Fast!)
        # ========================================
        # Simple: reward velocity when on road
        if state.on_road:
            speed_reward = (abs(state.velocity) / car.max_velocity) * 2.0
            reward += speed_reward

        # ========================================
        # 2.5. ANTICIPATORY BRAKING (Slow before walls!)
        # ========================================
        # Get observation to check front ray distance
        obs = self._get_observation()
        front_ray = obs[0]  # First value is front ray distance (0-1 normalized)

        # If wall is close ahead and speed is high, reward slowing down
        if front_ray < 0.4 and state.on_road:  # Wall within 40% of max ray distance
            speed_norm = abs(state.velocity) / car.max_velocity
            # Reward lower speed when approaching walls (teaches braking before corners!)
            anticipation_reward = (1.0 - speed_norm) * (0.4 - front_ray) * 2.0
            reward += anticipation_reward

        # ========================================
        # 3. RACING LINE (Stay on optimal path)
        # ========================================
        # Get distance to centerline
        dist_to_center, _ = self._get_distance_to_centerline(state.x, state.y)
        track_half_width = track.road_width / 2

        # Reward being close to centerline (racing line)
        if state.on_road:
            # Closer to center = better (0 = perfect, 1 = edge)
            centerline_score = 1.0 - (dist_to_center / track_half_width)
            reward += centerline_score * 0.5

        # ========================================
        # 4. OFF-ROAD PENALTY (speed-dependent)
        # ========================================
        if not state.on_road:
            # Worse penalty if going off-road at high speed (teaches braking!)
            speed_factor = abs(state.velocity) / car.max_velocity
            reward -= 2.0 * (1.0 + speed_factor)  # -2 to -4 depending on speed

            # OFF-ROAD RECOVERY: reward getting closer to road
            # This creates a gradient that guides the agent back
            if dist_to_center < track_half_width * 1.5:  # Still somewhat close
                # Reward being closer to centerline even when off-road
                recovery_reward = (1.5 - dist_to_center / track_half_width) * 0.3
                reward += recovery_reward

        # ========================================
        # 5. STUCK OFF-ROAD PENALTY
        # ========================================
        if self.consecutive_offroad_steps > 290:
            # About to terminate - heavy penalty
            reward -= 10.0
        if self.consecutive_offroad_steps >= 300:
            # Episode ending due to stuck
            reward -= 50.0

        # ========================================
        # 6. LAP COMPLETION BONUS
        # ========================================
        if state.lap_complete:
            # Big reward for finishing + time bonus
            base_reward = 500.0
            time_bonus = max(0.0, 500.0 - state.lap_time * 2.0)
            reward += base_reward + time_bonus

        return reward

    def _check_truncation(self, state) -> bool:
        #Check if episode should be truncated
        if self.current_step >= self.max_steps:
            return True

        # Truncate if no progress for too long
        if self.no_progress_steps > 500:
            return True

        # Truncate if stuck off-road for 5 seconds (300 steps at 60 FPS)
        if self.consecutive_offroad_steps > 300:
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
            self.engine.render()
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

    def close(self):
        self.engine.quit()


def make_env(render_mode: Optional[str] = None, max_steps: int = 3000):
    """
    Factory function to create the environment.

    Args:
        render_mode: "human" for visual rendering, None for headless
        max_steps: Maximum steps per episode

    Returns:
        RacingEnv instance
    """
    return RacingEnv(render_mode=render_mode, max_steps=max_steps)
