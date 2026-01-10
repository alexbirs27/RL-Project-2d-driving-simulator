from enum import Enum


class Action(Enum):
    """Possible actions for the car."""
    NONE = 0
    ACCELERATE = 1
    BRAKE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4