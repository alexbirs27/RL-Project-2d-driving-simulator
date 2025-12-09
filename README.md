# 2D Car Racing Simulator

A modular 2D top-down racing game built with Pygame, designed for future reinforcement learning integration.

## Requirements

- Python 3.8+
- Pygame

## Installation

```bash
pip install pygame
```

## Running the Game

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| UP | Accelerate |
| DOWN | Brake |
| LEFT | Turn left |
| RIGHT | Turn right |
| R | Restart lap |
| ESC | Quit |

## Project Structure

```
├── main.py          # Entry point
├── engine.py        # Game engine (orchestrates all components)
├── car.py           # Car physics and state
├── track.py         # Track definition and collision detection
├── renderer.py      # All rendering logic
├── actions.py       # Action enum for inputs
└── README.md
```

## Architecture

The codebase is designed for easy RL integration:

### Headless Mode

Run simulation without rendering (for training):

```python
from engine import GameEngine
from actions import Action

engine = GameEngine(render=False)
engine.init()

# Simulation loop
state = engine.step([Action.ACCELERATE], dt=1/60)
```

### State Access

The `CarState` dataclass exposes:

- `x`, `y` - Position
- `angle` - Rotation in radians
- `velocity` - Current speed
- `on_road` - Whether car is on track
- `lap_complete` - Whether lap is finished
- `lap_time` - Current lap time

### Separation of Concerns

- `engine.step()` - Physics update + collision detection
- `engine.render()` - Drawing (can be skipped)
- `engine.get_state()` - Get current observation

## Physics

- Friction slows the car naturally
- Off-road (grass) increases friction and limits max speed
- Turning speed scales with velocity

## Future RL Integration

Replace keyboard input with agent actions:

```python
from engine import GameEngine
from actions import Action

engine = GameEngine(render=False)
engine.init()

while not done:
    # Agent selects action
    action = agent.select_action(state)

    # Step environment
    state = engine.step([action], dt=1/60)

    # Calculate reward based on state
    reward = calculate_reward(state)
```
