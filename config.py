"""
Centralized configuration for the 2D racing RL project.

Change values here instead of hunting through multiple files!
"""

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

# Observation space
OBS_DIM = 10  # 5 rays + 2 che ckpoint nav + 3 racing line
RAY_MAX_DISTANCE = 400.0  # Pixels - how far agent can "see"
CHECKPOINT_NAV_MAX_DISTANCE = 600.0  # Max distance for normalizing checkpoint distance

# Ray configuration
RAY_ANGLES = [
    0.0,           # Front
    0.7854,        # Front-right (45° = π/4)
    -0.7854,       # Front-left (-45°)
    1.5708,        # Right (90° = π/2)
    -1.5708,       # Left (-90°)
]

# Episode limits
MAX_STEPS_PER_EPISODE = 12000
NO_PROGRESS_LIMIT = 1000  # Steps without checkpoint progress
OFFROAD_TRUNCATION_LIMIT = 600  # Steps off-road before truncation (10 seconds at 60 FPS)

# ============================================
# REWARD FUNCTION WEIGHTS
# ============================================

REWARD_CHECKPOINT = 50.0      # Per checkpoint reached (forward only)
REWARD_SPEED = 0.5            # Per step, scaled by velocity
REWARD_CENTERLINE = 0.5       # Per step, scaled by centerline proximity
REWARD_OFFROAD = -5.0         # Per step off-road
REWARD_LAP_COMPLETE = 1000.0   # Bonus for finishing lap

# ============================================
# DQN AGENT HYPERPARAMETERS
# ============================================

# Network architecture
ACTION_DIM = 9  # 9 discrete actions (nothing, accel, brake, turn L/R, combinations)

# Training hyperparameters
DQN_GAMMA = 0.99              # Discount factor
DQN_LEARNING_RATE = 5e-5      # Adam learning rate (more conservative for stability)
DQN_BUFFER_SIZE = 30000       # Replay buffer capacity
DQN_BATCH_SIZE = 128          # Minibatch size
DQN_TAU = 0.005               # Soft update rate for target network (faster target updates)

# Exploration parameters
DQN_EPSILON_START = 1.0       # Initial exploration rate
DQN_EPSILON_END = 0.05        # Final exploration rate (lower = more exploitation)
DQN_EPSILON_DECAY = 80000     # Decay steps (higher = slower decay, more exploration)

# Training settings
DQN_TOTAL_EPISODES = 1000     # Total training episodes

# ============================================
# TRACK CONFIGURATION
# ============================================

# F1Tenth track loading
TRACK_SCALE = 50.0            # Pixels per meter
TRACK_ROAD_WIDTH = 100.0      # Track width in pixels
CHECKPOINT_SPACING = 12       # Every Nth point becomes a checkpoint (higher = fewer checkpoints)
# Result: ~60-72 strategic checkpoints from 864 centerline points

# ============================================
# CAR PHYSICS PARAMETERS
# ============================================

CAR_WIDTH = 15
CAR_HEIGHT = 30

CAR_MAX_VELOCITY = 200.0
CAR_ACCELERATION = 150.0
CAR_BRAKE_FORCE = 140.0
CAR_FRICTION = 30.0
CAR_TURN_SPEED = 3.0

# Drift physics (RWD simulation)
CAR_GRIP_THRESHOLD = 150.0      # Speed at which rear loses grip
CAR_LATERAL_FRICTION = 120.0    # How quickly drift is reduced
CAR_DRIFT_STRENGTH = 5.0        # How much the rear kicks out

# Off-road physics
CAR_OFFROAD_FRICTION_MULT = 2.5
CAR_OFFROAD_MAX_VELOCITY = 60.0

# ============================================
# RENDERING CONFIGURATION
# ============================================

RENDER_WIDTH = 1600
RENDER_HEIGHT = 1200
RENDER_FPS = 60
RENDER_ZOOM = 1.5  # Camera zoom level (higher = closer)

# Colors (RGB)
COLOR_GRASS = (34, 139, 34)
COLOR_ROAD = (60, 60, 60)
COLOR_ROAD_EDGE = (255, 255, 255)
COLOR_CAR = (255, 0, 0)
COLOR_FINISH_LINE = (255, 255, 0)

# Ray colors for visualization
RAY_COLORS = [
    (0, 255, 0),      # Front: GREEN
    (255, 255, 0),    # Front-right: YELLOW
    (0, 255, 255),    # Front-left: CYAN
    (255, 165, 0),    # Right: ORANGE
    (138, 43, 226),   # Left: PURPLE
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_config_summary():
    """Print a summary of the current configuration."""
    return f"""
=== RL Racing Configuration ===

Observation Space: {OBS_DIM} values
Ray Distance: {RAY_MAX_DISTANCE}px
Max Episode Steps: {MAX_STEPS_PER_EPISODE}

DQN Hyperparameters:
  Learning Rate: {DQN_LEARNING_RATE}
  Epsilon: {DQN_EPSILON_START} → {DQN_EPSILON_END} (decay: {DQN_EPSILON_DECAY})
  Batch Size: {DQN_BATCH_SIZE}
  Buffer Size: {DQN_BUFFER_SIZE}

Rewards:
  Checkpoint: +{REWARD_CHECKPOINT}
  Speed: +{REWARD_SPEED}/step
  Centerline: +{REWARD_CENTERLINE}/step
  Off-road: {REWARD_OFFROAD}/step
  Lap Complete: +{REWARD_LAP_COMPLETE}

Car Physics:
  Max Speed: {CAR_MAX_VELOCITY}
  Acceleration: {CAR_ACCELERATION}
  Turn Speed: {CAR_TURN_SPEED}

================================
"""

if __name__ == "__main__":
    print(get_config_summary())
