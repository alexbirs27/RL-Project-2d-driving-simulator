import numpy as np
import math
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

from engine import GameEngine
from actions import Action


class RacingEnv(gym.Env):
    """
    Gymnasium environment wrapper for the 2D racing game.
    
    Observation space:
        - Car velocity (normalized)
        - Car angle (sin and cos)
        - Distance sensors (8 directions)
        - Progress on track (0-1)
        - On road flag
    
    Action space:
        - 0: No action
        - 1: Accelerate
        - 2: Brake
        - 3: Turn left
        - 4: Turn right
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, max_steps=3000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize game engine
        self.engine = GameEngine(width=1200, height=800, render=(render_mode == 'human'))
        self.engine.init()
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: velocity(1) + angle(2) + sensors(8) + progress(1) + on_road(1) = 13
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )
        
        # Tracking variables
        self.previous_progress = 0.0
        self.total_reward = 0.0
        self.dt = 1.0 / 60.0  # Fixed timestep
        
    def _get_observation(self) -> np.ndarray:
        """Generate observation vector from current game state."""
        state = self.engine.get_state()
        
        # Normalize velocity (-1 to 1)
        velocity_norm = state.velocity / self.engine.car.max_velocity
        
        # Angle as sin/cos (handles circular nature)
        angle_sin = math.sin(state.angle)
        angle_cos = math.cos(state.angle)
        
        # Distance sensors (8 directions)
        sensors = self._get_distance_sensors()
        
        # Progress on track (0-1)
        progress = self._calculate_progress()
        
        # On road flag
        on_road = 1.0 if state.on_road else -1.0
        
        observation = np.array([
            velocity_norm,
            angle_sin,
            angle_cos,
            *sensors,
            progress,
            on_road
        ], dtype=np.float32)
        
        return observation
    
    def _get_distance_sensors(self, num_sensors=8, max_distance=200.0) -> list:
        """
        Cast rays in multiple directions to detect distance to track edges.
        Returns normalized distances (0=max_distance, 1=very close).
        """
        state = self.engine.get_state()
        sensors = []
        
        for i in range(num_sensors):
            angle = state.angle + (2 * math.pi * i / num_sensors)
            distance = self._cast_ray(state.x, state.y, angle, max_distance)
            # Normalize: closer = higher value
            normalized = 1.0 - (distance / max_distance)
            sensors.append(normalized)
        
        return sensors
    
    def _cast_ray(self, x: float, y: float, angle: float, max_distance: float) -> float:
        """Cast a ray and find distance to track edge."""
        step_size = 5.0
        distance = 0.0
        
        dx = math.sin(angle) * step_size
        dy = -math.cos(angle) * step_size
        
        current_x, current_y = x, y
        
        while distance < max_distance:
            current_x += dx
            current_y += dy
            distance += step_size
            
            if not self.engine.track.is_on_road(current_x, current_y):
                return distance
        
        return max_distance
    
    def _calculate_progress(self) -> float:
        """Calculate how far along the track the car is (0-1)."""
        state = self.engine.get_state()
        track_points = self.engine.track.center_points
        
        # Find closest track point
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(track_points):
            dist = math.sqrt((state.x - point[0])**2 + (state.y - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Progress is the index normalized to [0, 1]
        return closest_idx / len(track_points)
    
    def _calculate_reward(self, state, prev_progress: float) -> Tuple[float, bool]:
        """
        Calculate reward based on:
        - Progress on track
        - Speed (faster is better when on road)
        - Staying on road
        - Completing lap
        """
        reward = 0.0
        terminated = False
        
        # Progress reward (most important)
        current_progress = self._calculate_progress()
        progress_delta = current_progress - prev_progress
        
        # Handle wrap-around at finish line
        if progress_delta < -0.5:
            progress_delta += 1.0
        
        reward += progress_delta * 100.0  # Scale progress reward
        self.previous_progress = current_progress
        
        # Speed reward (when on road)
        if state.on_road:
            speed_reward = (state.velocity / self.engine.car.max_velocity) * 0.5
            reward += speed_reward
        else:
            # Penalty for being off road
            reward -= 1.0
        
        # Lap completion bonus
        if state.lap_complete:
            reward += 1000.0
            terminated = True
        
        # Time penalty (encourage faster completion)
        reward -= 0.1
        
        # Penalize very low speeds (stuck)
        if abs(state.velocity) < 5.0 and self.current_step > 60:
            reward -= 0.5
        
        return reward, terminated
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.engine.reset()
        self.current_step = 0
        self.previous_progress = 0.0
        self.total_reward = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Map discrete action to game action
        action_map = {
            0: [],
            1: [Action.ACCELERATE],
            2: [Action.BRAKE],
            3: [Action.TURN_LEFT],
            4: [Action.TURN_RIGHT],
        }
        
        actions = action_map[action]
        
        # Store previous progress for reward calculation
        prev_progress = self.previous_progress
        
        # Execute action in game
        state = self.engine.step(actions, self.dt)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward, terminated = self._calculate_reward(state, prev_progress)
        self.total_reward += reward
        
        # Check if truncated (max steps reached)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'lap_time': state.lap_time,
            'velocity': state.velocity,
            'on_road': state.on_road,
            'progress': self._calculate_progress(),
            'total_reward': self.total_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self.engine.render()
    
    def close(self):
        """Clean up resources."""
        self.engine.quit()