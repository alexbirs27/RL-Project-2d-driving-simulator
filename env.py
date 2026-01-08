"""
Gymnasium-compatible environment wrapper for the racing game.

This environment can be used by ALL RL algorithms (DQN, PPO, A2C, etc.)

Observation Space (13 values):
    - velocity (normalized)
    - lateral_velocity (drift, normalized)  # NEW
    - angular velocity approximation
    - on_road flag
    - distance to track center (normalized)
    - direction to next checkpoint (cos)
    - direction to next checkpoint (sin)
    - angle difference to track direction
    - distance to next checkpoint (normalized)
    - car direction (cos)
    - car direction (sin)
    - track progress
    - speed as fraction of max

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
    - +10.0 per new checkpoint visited
    - +0.5 for speed (when on road)
    - +0.1 for staying on road
    - -1.0 for going off road
    - -0.3 for standing still
    - +0.05 for accelerating (encourages movement)
    - +200 + time_bonus for completing lap (faster = better)
    - -0.05 per step (time penalty)

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
            width=1200,
            height=800,
            render=(render_mode == "human")
        )

        # 9 actions: combinations of acceleration/brake with steering
        self.action_space = spaces.Discrete(9)

        # 13 observations (added lateral_velocity for drift)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )

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
        self.last_angle = 0.0
        self.no_progress_steps = 0

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
        self.last_angle = self.engine.car.angle
        self.no_progress_steps = 0
        self._last_dist_to_next = None  # Reset distance tracker

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Execute one environment step.
        actions = self.action_map[action]
        state = self.engine.step(actions, self.dt)
        self.current_step += 1

        obs = self._get_observation()
        reward = self._calculate_reward(state, action)
        terminated = state.lap_complete
        truncated = self._check_truncation(state)
        info = self._get_info()

        self.last_angle = state.angle

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Convert game state to observation array
        state = self.engine.get_state()
        car = self.engine.car
        track = self.engine.track

        # Current checkpoint and next checkpoint
        current_cp = track.get_nearest_checkpoint(state.x, state.y)
        next_cp = (current_cp + 3) % track.num_checkpoints  # Look a few checkpoints ahead

        # Direction to next checkpoint
        next_point = track.center_points[next_cp]
        dx = next_point[0] - state.x
        dy = next_point[1] - state.y
        dist_to_next = math.sqrt(dx * dx + dy * dy)

        # Normalize direction to next checkpoint
        if dist_to_next > 0:
            dir_to_next_x = dx / dist_to_next
            dir_to_next_y = dy / dist_to_next
        else:
            dir_to_next_x = 0.0
            dir_to_next_y = 0.0

        # Angle to next checkpoint
        angle_to_next = math.atan2(dx, -dy)  # Same convention as car angle

        # Angle difference (how much we need to turn)
        angle_diff = angle_to_next - state.angle
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Car direction
        car_dir_x = math.sin(state.angle)
        car_dir_y = -math.cos(state.angle)

        # Distance to track center
        center_point = track.center_points[current_cp]
        dist_to_center = math.sqrt(
            (state.x - center_point[0]) ** 2 +
            (state.y - center_point[1]) ** 2
        )
        dist_to_center_norm = np.clip(dist_to_center / (track.road_width / 2), 0, 2) - 1

        # Angular velocity approximation
        angular_vel = (state.angle - self.last_angle) / self.dt
        angular_vel_norm = np.clip(angular_vel / 3.0, -1, 1)

        # Progress
        progress = track.get_progress(current_cp)

        # Normalize lateral velocity (drift)
        lateral_vel_norm = np.clip(state.lateral_velocity / 100.0, -1, 1)

        obs = np.array([
            np.clip(state.velocity / car.max_velocity, -1, 1),          # Velocity normalized
            lateral_vel_norm,                                            # NEW: Lateral velocity (drift)
            angular_vel_norm,                                            # Angular velocity
            1.0 if state.on_road else -1.0,                             # On road
            dist_to_center_norm,                                         # Distance to center
            dir_to_next_x,                                               # Direction to next CP (x)
            dir_to_next_y,                                               # Direction to next CP (y)
            np.clip(angle_diff / math.pi, -1, 1),                       # Angle difference
            np.clip(dist_to_next / 300.0, 0, 1) * 2 - 1,               # Distance to next CP
            car_dir_x,                                                   # Car direction (x)
            car_dir_y,                                                   # Car direction (y)
            progress * 2 - 1,                                            # Progress
            np.clip(state.velocity / car.max_velocity, 0, 1),           # Speed fraction
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, state, action: int) -> float:
        # Calculate reward for the current step
        reward = 0.0
        track = self.engine.track
        car = self.engine.car

        current_cp = track.get_nearest_checkpoint(state.x, state.y)

        # Big reward for visiting new checkpoints
        if current_cp not in self.visited_checkpoints:
            # Check if it's forward progress (not going backwards)
            diff = (current_cp - self.last_checkpoint) % track.num_checkpoints

            if diff > 0 and diff < track.num_checkpoints // 2:
                reward += 20.0 * diff  # Increased reward for forward progress
                self.visited_checkpoints.add(current_cp)
                self.no_progress_steps = 0
            elif diff > track.num_checkpoints // 2:
                # Going backwards
                reward -= 2.0
                self.no_progress_steps += 1
        else:
            self.no_progress_steps += 1

        self.last_checkpoint = current_cp

        # === NEW: Reward for facing towards next checkpoint ===
        next_cp = (current_cp + 3) % track.num_checkpoints
        next_point = track.center_points[next_cp]
        dx = next_point[0] - state.x
        dy = next_point[1] - state.y
        dist_to_next = math.sqrt(dx * dx + dy * dy)

        if dist_to_next > 0:
            # Direction to next checkpoint (normalized)
            dir_to_next_x = dx / dist_to_next
            dir_to_next_y = dy / dist_to_next

            # Car's facing direction
            car_dir_x = math.sin(state.angle)
            car_dir_y = -math.cos(state.angle)

            # Alignment: dot product (1 = facing perfectly, -1 = facing away)
            alignment = dir_to_next_x * car_dir_x + dir_to_next_y * car_dir_y

            # Reward for good alignment (0 to 0.3 based on how well aligned)
            if alignment > 0:
                reward += alignment * 0.3

        # Reward for getting closer to next checkpoint ===
        if self._last_dist_to_next is not None:
            dist_improvement = self._last_dist_to_next - dist_to_next
            if dist_improvement > 0:
                reward += dist_improvement * 0.02  # Small reward for getting closer
        self._last_dist_to_next = dist_to_next

        # Reward for speed (only when on road and moving forward)
        if state.on_road and state.velocity > 0:
            speed_reward = (state.velocity / car.max_velocity) * 1.0
            reward += speed_reward

        # Penalty for being off road
        if not state.on_road:
            reward -= 0.5

        # Reward for staying on road
        if state.on_road:
            reward += 0.2

        # Penalty for standing still
        if abs(state.velocity) < 5.0:
            reward -= 0.1

        # Reward for accelerating (encourage movement)
        if action in [1, 5, 6]:  # Actions with acceleration
            reward += 0.1

        # Big reward for completing lap - faster = better!
        if state.lap_complete:
            base_reward = 500.0  # Increased base reward
            # Time bonus: faster lap = more reward
            time_bonus = max(0.0, 1000.0 - state.lap_time * 10.0)
            reward += base_reward + time_bonus

        return reward

    def _check_truncation(self, state) -> bool:
        #Check if episode should be truncated
        if self.current_step >= self.max_steps:
            return True

        # Truncate if no progress for too long
        if self.no_progress_steps > 500:
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
