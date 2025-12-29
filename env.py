"""
Custom Gymnasium environment for the 2D driving simulator.

Observation Space (8 features):
    - x_norm: Normalized x position [0, 1]
    - y_norm: Normalized y position [0, 1]
    - angle_sin: Sine of car angle [-1, 1]
    - angle_cos: Cosine of car angle [-1, 1]
    - velocity_norm: Normalized velocity [0, 1]
    - on_road: Whether car is on road (0 or 1)
    - distance_to_center_norm: Normalized distance to track center [0, 1]
    - track_progress: Progress around the track [0, 1]

Action Space (Discrete 5):
    0: NONE
    1: ACCELERATE
    2: BRAKE
    3: TURN_LEFT
    4: TURN_RIGHT

Reward Structure:
    - Speed reward: Positive reward for moving fast on road
    - Off-road penalty: Negative reward when off the track
    - Lap completion: Large positive reward for completing a lap
    - Time penalty: Small negative reward per step to encourage efficiency


Usage: from env import DrivingEnv, make_env
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from engine import GameEngine
from actions import Action


class DrivingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,  # "human" for visual rendering, None for headless
        max_steps: int = 3000,  # Maximum steps before truncation (default ~50 seconds)
        dt: float = 1/60  # Time step in seconds (default 1/60 for 60 FPS)
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt

        # Create game engine (headless unless render_mode is "human")
        self._render_enabled = render_mode == "human"
        self.engine = GameEngine(render=self._render_enabled)

        # Define observation space (8 continuous features)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)

        # Action mapping
        self._action_map = {
            0: [],                      # NONE
            1: [Action.ACCELERATE],     # ACCELERATE
            2: [Action.BRAKE],          # BRAKE
            3: [Action.TURN_LEFT],      # TURN_LEFT
            4: [Action.TURN_RIGHT],     # TURN_RIGHT
        }

        # Episode tracking
        self.current_step = 0
        self.prev_progress = 0.0
        self._initialized = False

    def _init_pygame(self):
        if not self._initialized:
            self.engine.init()
            self._initialized = True

    def _get_observation(self) -> np.ndarray:
        # Convert car state to observation vector
        # Returns: 8-dimensional numpy array with normalized features
        state = self.engine.get_state()

        # Normalize position to [0, 1]
        x_norm = state.x / self.engine.width
        y_norm = state.y / self.engine.height

        # Use sin/cos for angle (avoids discontinuity at -pi/pi)
        angle_sin = math.sin(state.angle)
        angle_cos = math.cos(state.angle)

        # Normalize velocity to [0, 1]
        max_vel = self.engine.car.max_velocity
        velocity_norm = np.clip(state.velocity / max_vel, 0, 1)

        # On-road flag
        on_road = 1.0 if state.on_road else 0.0

        # Distance to track center (normalized)
        distance_to_center = self._get_distance_to_center(state.x, state.y)
        distance_norm = np.clip(distance_to_center / (self.engine.track.road_width / 2), 0, 1)

        # Track progress [0, 1]
        track_progress = self._get_track_progress(state.x, state.y)

        return np.array([
            x_norm,
            y_norm,
            angle_sin,
            angle_cos,
            velocity_norm,
            on_road,
            distance_norm,
            track_progress
        ], dtype=np.float32)

    def _get_distance_to_center(self, x: float, y: float) -> float:
        # Calculate distance from point to nearest track center
        min_dist = float('inf')
        track = self.engine.track

        for i in range(len(track.center_points)):
            p1 = track.center_points[i]
            p2 = track.center_points[(i + 1) % len(track.center_points)]
            dist = track._point_to_segment_distance(x, y, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _get_track_progress(self, x: float, y: float) -> float:
        # Calculate progress around the track as a value from 0 to 1
        # Returns: Float from 0 to 1 indicating progress around the track
        track = self.engine.track
        min_dist = float('inf')
        closest_segment = 0

        for i in range(len(track.center_points)):
            p1 = track.center_points[i]
            p2 = track.center_points[(i + 1) % len(track.center_points)]
            dist = track._point_to_segment_distance(x, y, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_segment = i

        # Progress is the segment index divided by total segments
        progress = closest_segment / len(track.center_points)
        return progress

    def _calculate_reward(self, state, prev_progress: float) -> float:
        reward = 0.0

        # Speed reward (only when on road)
        if state.on_road:
            speed_reward = (state.velocity / self.engine.car.max_velocity) * 0.1
            reward += speed_reward

        # Progress reward
        current_progress = self._get_track_progress(state.x, state.y)

        # Handle wrap-around (when progress goes from ~1 to ~0)
        progress_delta = current_progress - prev_progress
        if progress_delta < -0.5:  # Wrapped around
            progress_delta += 1.0
        elif progress_delta > 0.5:  # Went backwards across start
            progress_delta -= 1.0

        reward += progress_delta * 10.0  # Scale progress reward

        # Off-road penalty
        if not state.on_road:
            reward -= 0.5

        # Lap completion bonus
        if state.lap_complete:
            # Bonus inversely proportional to lap time (faster = more reward)
            time_bonus = max(100 - state.lap_time, 50)
            reward += time_bonus

        # Small time penalty to encourage efficiency
        reward -= 0.01

        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Returns: observation: Initial observation, info: Additional information dict
        super().reset(seed=seed)

        # Initialize pygame if needed
        self._init_pygame()

        # Reset the game engine
        self.engine.reset()

        # Reset episode tracking
        self.current_step = 0
        self.prev_progress = 0.0

        observation = self._get_observation()
        info = {"lap_time": 0.0, "on_road": True}

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Returns: new observation after action, reward for this step,
        # terminated: Whether episode ended (lap complete),
        # truncated: Whether episode was cut short (max steps), additional information
        self.current_step += 1

        # Convert action to engine format
        actions = self._action_map.get(action, [])

        # Store previous progress for reward calculation
        prev_progress = self._get_track_progress(
            self.engine.car.x,
            self.engine.car.y
        )

        # Execute step in game engine
        state = self.engine.step(actions, self.dt)

        # Calculate reward
        reward = self._calculate_reward(state, prev_progress)

        # Check termination conditions
        terminated = state.lap_complete
        truncated = self.current_step >= self.max_steps

        # Get observation
        observation = self._get_observation()

        # Build info dict
        info = {
            "lap_time": state.lap_time,
            "on_road": state.on_road,
            "velocity": state.velocity,
            "lap_complete": state.lap_complete,
            "step": self.current_step
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.engine.render()

    def close(self):
        if self._initialized:
            self.engine.quit()
            self._initialized = False


def make_env(render_mode: Optional[str] = None, max_steps: int = 3000) -> DrivingEnv:
    return DrivingEnv(render_mode=render_mode, max_steps=max_steps)


# For compatibility with standard gym.make() pattern
def register_env():
    # Register the environment with Gymnasium.
    from gymnasium.envs.registration import register

    register(
        id="DrivingSimulator-v0",
        entry_point="env:DrivingEnv",
        max_episode_steps=3000,
    )
