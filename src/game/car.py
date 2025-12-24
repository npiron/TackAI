
from dataclasses import dataclass
import numpy as np

@dataclass
class Car:
    pos: np.ndarray
    vel: np.ndarray
    heading: float
    speed: float
    steering: float = 0.0
    angular_velocity: float = 0.0
    
    # Drift mechanics
    drift_angle: float = 0.0  # Angle entre heading et velocity
    is_drifting: bool = False
    drift_time: float = 0.0
    
    # Stats tracking
    max_speed_reached: float = 0.0
