# DQN Training Log

## FIRST RUN: Initial Implementation

**Environment**: Simple 8 states, 5 actions

**Observations**: Agent not moving from start, rewards very negative (-1500 to -3000), no improvement.
Environment too simple, agent couldn't sense direction to next checkpoint. Hard target updates caused instability.

**Optimizations**: Upgraded to 12 states and 9 combined actions (accelerate+turn), applied soft updates (τ=0.005), gradient clipping, Huber loss, exponential epsilon decay.

---

## SECOND RUN: Stable Learning

**Results**: Agent learns around episode 45-50, converges to stable 4400+ reward. Lap completion rate reaches ~95%, lap times consistent at ~40s. Much faster learning than PPO, more stable convergence. Environment appears too easy - agent masters it completely.

**Next Step**: Make environment harder (drift physics, bigger/complex track, obstacles).

---

## THIRD RUN: Drift Physics Integration

**Environment Modifications**:
- Observation space expanded from 12 to 13 dimensions (added lateral_velocity)
- Implemented RWD drift mechanics: grip_threshold=150.0, drift_strength=5.0, lateral_friction=120.0

**Training Results (50 episodes)**:
- Episodes 0-10: Rapid convergence, mean reward 1268 → 4358
- Episodes 10-49: Stable performance at 4380±10 reward
- Final lap completion: 100%, lap time improved 45.90s → 13.48s (70.6% reduction)
- Off-road rate: 79% → 17%

**Anomalies**: Episodes 27-29 showed temporary degradation, recovered within 1 episode (exploration event, not catastrophic forgetting).

**Conclusion**: DQN adapted to drift physics within 10 episodes. Enhanced observation space facilitated learning.

---

## FOURTH RUN: Complex Track (CATASTROPHIC FAILURE)

**Environment**: Rounded rectangle track, 138 checkpoints, dynamic camera following

**Training Results (300 episodes)**:
- **Catastrophic forgetting** - performance collapsed suddenly mid-training
- Lap completion peaked at 43%, degraded to 5% by end
- Off-road rate: 40-50% (no improvement)
- Critical failure: Epsilon reached minimum by episode 24 (premature exploration decay)

**Analysis**: Agent showed initial learning followed by severe policy degradation. Epsilon-greedy exploration ceased too early (within first 8% of training), preventing recovery from performance collapse.

**Conclusion**: DQN failed due to premature exploration decay and catastrophic forgetting.

**Recommended Fixes**: Slower epsilon decay, higher minimum epsilon, reduced learning rate, curriculum learning.

---

## FIFTH RUN: F1Tenth Real Track + Observation Redesign

**Environment Modifications**:
- Real F1Tenth Spielberg track (864 centerline points, complex geometry)
- Observation space: 12D → 8D hybrid approach
  - 5 raycasts (collision detection): front, front-right, front-left, right, left
  - 2 centerline metrics (racing line): distance + angle to centerline
  - 1 velocity reading
- Removed broken lookahead implementation (offsets [30,60,90] were only 3-15m ahead on 864-point track)

**Initial Training (134 episodes) - CATASTROPHIC FORGETTING**:
- Episodes 80-90: Mean reward collapsed 2287 → 943 (60% drop)
- Episodes 90-100: Further collapse 943 → 168 (82% drop)
- Off-road rate increased from 65% → 70%
- Agent learned then unlearned, systematic performance degradation

**Root Cause Analysis**:
1. **No anticipatory braking**: Speed reward only triggered when on road, no signal to slow before corners
2. **No off-road recovery**: Flat -2 penalty regardless of direction, centerline reward only worked on-road → agent drifted farther away when off-track

**Reward Function Improvements**:
- **Speed-dependent off-road penalty**: -2 to -4 based on velocity (teaches braking before corners)
- **Off-road recovery gradient**: Reward getting closer to centerline when off-road (guides agent back to track)
- **Anticipatory braking**: Reward slowing down when front ray detects walls (brake before corners, not after crash)

**Next**: Retrain with improved reward system, expect agent to learn corner braking and off-road recovery behaviors.
