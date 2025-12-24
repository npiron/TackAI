
import math
import numpy as np
import random
from typing import Optional, List, Tuple

from src.core.constants import *
from src.game.track import Track, generate_procedural_track
from src.game.car import Car
from src.utils.math_utils import vec
from src.utils.jit_ops import cast_rays_jit

class TimeAttackEnv:
    """
    Minimal env:
      obs = 14 floats (Pilot-Centric):
            [speed_n, local_vx, local_vy, ang_vel, prev_steer, 
             target_dist, target_dir_x, target_dir_y, on_track, 
             lidar_1, lidar_2, lidar_3, lidar_4, lidar_5]
      action = (steer, accel, brake) in [-1..1], [0..1], [0..1]
    """
    def __init__(self, track: Optional[Track] = None, procedural: bool = True):
        self.procedural = procedural
        if track:
            self.track = track
            self.procedural = False # Force static if track provided explicitly
        else:
            self.track = generate_procedural_track()
            
        self.car: Optional[Car] = None

        self.t = 0.0
        self.done = False

        self.best_time: Optional[float] = None
        self.best_ghost: List[Tuple[float, float]] = []

        self.ghost: List[Tuple[float, float]] = []
        self.cp_visited = [False] * len(self.track.checkpoints)
        self.off_track_counter = 0  # NEW: Track how long car is off-road

        self.reset()

    def reset(self):
        # Regenerate track if procedural mode is active
        if self.procedural:
            self.track = generate_procedural_track()
            # print(f"DEBUG: Track generated with {len(self.track.checkpoints)} CPs")
            
        sx, sy = self.track.start_pos
        self.car = Car(
            pos=vec(sx, sy),
            vel=vec(0, 0),
            heading=self.track.start_heading,
            speed=0.0
        )
        self.t = 0.0
        self.done = False
        self.cp_visited = [False] * len(self.track.checkpoints)
        self.ghost = []
        self.off_track_counter = 0  # Reset timeout counter
        self.prev_steer = 0.0       # Track previous action
        
        return self._obs()

    def step(self, action: Tuple[float, float, float]):
        # action: steer, accel, brake
        steer, accel, brake = action
        
        # ========== REALISTIC CAR PHYSICS ==========
        
        # 1. Update Speed (longitudinal)
        if accel > 0:
            self.car.speed += accel * ACCEL
        if brake > 0:
            self.car.speed -= brake * BRAKE
            
        # Friction / Drag (air resistance)
        self.car.speed *= FRICTION
        
        # Cap speed
        self.car.speed = max(0.0, min(self.car.speed, MAX_SPEED))
        
        # 2. Physics-based Steering (Inertia & Virtual Steering Wheel)
        # Smoothly update the car's steering angle based on input (Arcade Feel)
        STEER_SPEED = 0.15 # 1.0 / 0.15 = ~7 frames to full lock
        
        # Update steering value
        if steer > self.car.steering:
            self.car.steering = min(steer, self.car.steering + STEER_SPEED)
        elif steer < self.car.steering:
            self.car.steering = max(steer, self.car.steering - STEER_SPEED)
            
        # Instead of direct heading change, we change angular velocity
        # Steering applies torque (rotational acceleration)
        TURN_POWER = 0.05  # How fast we can change turning rate
        target_av = self.car.steering * TURN_SPEED # Max turn speed based on smoothed steering
        
        # Smoothly interpolate angular velocity towards target
        self.car.angular_velocity += (target_av - self.car.angular_velocity) * TURN_POWER
        
        # Speed factor still applies (harder to turn at high speed)
        speed_factor = 1.0 - (self.car.speed / MAX_SPEED) * 0.3
        
        # Apply rotation
        self.car.heading += self.car.angular_velocity * speed_factor
        
        # Store action for next obs
        self.prev_steer = steer
        
        # 3. Calculate intended velocity (where car WANTS to go)
        intended_vx = math.cos(self.car.heading) * self.car.speed
        intended_vy = math.sin(self.car.heading) * self.car.speed
        intended_vel = vec(intended_vx, intended_vy)
        
        # 4. Apply lateral friction (prevent sliding/drifting)
        # The car's velocity gradually aligns with its heading direction
        # This creates grip and prevents unrealistic serpentine movement
        
        # Lateral friction strength (0.0 = ice, 1.0 = instant grip)
        # Defined in constants.py (lower = more drift)
        
        # Gradually transition current velocity towards intended velocity
        self.car.vel = self.car.vel * (1.0 - LATERAL_GRIP) + intended_vel * LATERAL_GRIP
        
        # 5. Update Position
        new_pos = self.car.pos + self.car.vel
        
        # 4a. Check if new position would be off-road (HARD WALLS)
        # 4a. Check if new position would be off-road (HARD WALLS with SLIDING)
        if self.track.inside_track(new_pos):
            # Safe to move
            self.car.pos = new_pos
            self.off_track_counter = 0
        else:
            # WALL HIT! Try to slide.
            # 1. Try moving along X only
            pos_x_only = self.car.pos + vec(self.car.vel[0], 0)
            if self.track.inside_track(pos_x_only):
                self.car.pos = pos_x_only
                self.car.vel[1] = 0 # Kill Y velocity
                self.car.speed *= 0.9 # Friction
            else:
                # 2. Try moving along Y only
                pos_y_only = self.car.pos + vec(0, self.car.vel[1])
                if self.track.inside_track(pos_y_only):
                    self.car.pos = pos_y_only
                    self.car.vel[0] = 0 # Kill X velocity
                    self.car.speed *= 0.9 # Friction
                else:
                    # 3. Stuck -> Stop
                    self.car.speed *= 0.5
                    self.car.vel *= 0.5
            
            self.off_track_counter = 0 # No timeout game over

        # 4b. Check screen boundaries (backup safety)
        if self.track.hits_wall(self.car.pos):
            self.car.speed *= 0.3
            self.car.pos[0] = max(5, min(W - 5, self.car.pos[0]))
            self.car.pos[1] = max(5, min(H - 5, self.car.pos[1]))
        
        # 4c. BOOST ZONES - Speed boost!
        if self.track.on_boost(self.car.pos):
            self.car.speed = min(self.car.speed * 1.15, MAX_SPEED * 1.2)  # +15% speed, max 120%
        
        # 4d. DRIFT MECHANICS
        # Calculate drift angle (difference between heading and velocity direction)
        if self.car.speed > 0.1:
            vel_angle = np.arctan2(self.car.vel[1], self.car.vel[0])
            drift_angle = abs(self.car.heading - vel_angle)
            # Normalize to [0, pi]
            while drift_angle > np.pi:
                drift_angle -= 2 * np.pi
            drift_angle = abs(drift_angle)
            
            # Drifting if angle > 15° and turning
            is_drifting = drift_angle > 0.26 and abs(self.car.angular_velocity) > 0.01  # 15 degrees
            
            # Track max speed
            if not hasattr(self.car, 'max_speed_reached'):
                self.car.max_speed_reached = 0.0
            self.car.max_speed_reached = max(self.car.max_speed_reached, self.car.speed)

        # 5. Checkpoints
        self._update_checkpoints()
        
        # 6. Finish
        if self._check_finish():
            self.done = True
            if self.best_time is None or self.t < self.best_time:
                self.best_time = self.t
                self.best_ghost = list(self.ghost)

        # Record ghost
        if len(self.ghost) == 0 or np.linalg.norm(self.car.pos - vec(*self.ghost[-1])) > 5.0:
            self.ghost.append((self.car.pos[0], self.car.pos[1]))

        # Time
        self.t += 1.0 / FPS

        reward = 0.0
        # If done (win) -> Big Reward?
        # Typically handled in wrapper.
        if self.done:
            reward = 100.0

        return self._obs(), reward, self.done, {}

    def _update_checkpoints(self):
        # We need to enforce order? 
        # Yes, usually CP 1 then 2...
        # Next expected CP index
        next_cp_idx = 0
        for i, visited in enumerate(self.cp_visited):
            if not visited:
                next_cp_idx = i
                break
        else:
            # All visited
            return

        cx, cy, _ = self.track.checkpoints[next_cp_idx]
        dist = np.linalg.norm(self.car.pos - vec(cx, cy))
        if dist < CHECKPOINT_RADIUS:
            self.cp_visited[next_cp_idx] = True

    def _check_finish(self) -> bool:
        if not all(self.cp_visited):
            return False
        
        fx, fy = self.track.finish_pos[:2]
        dist = np.linalg.norm(self.car.pos - vec(fx, fy))
        return dist < FINISH_RADIUS

    def _obs(self):
        # PILOT-CENTRIC OBSERVATION (Local Coordinates)
        # We want the AI to learn "Driving", not "Map Memorization".
        
        # 1. Car Physics State
        speed_n = self.car.speed / MAX_SPEED
        
        # Local Velocity (Body Frame)
        # Project global velocity onto car's heading vectors
        head_x = math.cos(self.car.heading)
        head_y = math.sin(self.car.heading)
        
        # Global vel
        vx, vy = self.car.vel
        
        # Dot product
        # Forward vel: vel dot heading
        local_vx = (vx * head_x + vy * head_y) / MAX_SPEED
        # Lateral vel (Drift): vel dot (heading rotated 90 deg)
        # Rotated 90 deg vector is (-head_y, head_x)
        local_vy = (vx * -head_y + vy * head_x) / MAX_SPEED
        
        # Angular Velocity & Action History
        ang_vel = self.car.angular_velocity
        prev_steer = self.prev_steer
        # 2. Navigation (Next Checkpoint/Finish)
        next_obj = None
        for i, visited in enumerate(self.cp_visited):
            if not visited:
                next_obj = self.track.checkpoints[i][:2]
                break
        
        if next_obj is None:
            next_obj = self.track.finish_pos[:2]
            
        # Global vector to target
        tx, ty = next_obj
        dx_global = tx - self.car.pos[0]
        dy_global = ty - self.car.pos[1]
        
        dist = math.sqrt(dx_global**2 + dy_global**2)
        dist_n = min(1.0, dist / (W * 1.4)) # Normalize by approximate max diagonal
        
        # Local vector to target (Rotate global vector by -heading)
        # x' = x cos(-h) - y sin(-h)
        # y' = x sin(-h) + y cos(-h)
        # cos(-h) = cos(h), sin(-h) = -sin(h)
        
        target_local_x = dx_global * head_x + dy_global * head_y  # Forward distance
        target_local_y = -dx_global * head_y + dy_global * head_x # Rightward distance
        
        # Normalize target direction (Unit vector pointing to target in local frame)
        if dist > 0.1:
            tgt_dir_x = target_local_x / dist
            tgt_dir_y = target_local_y / dist
        else:
            tgt_dir_x, tgt_dir_y = 1.0, 0.0
            
        on_track = 1.0 if self.track.inside_track(self.car.pos) else 0.0

        # === LIDAR SENSORS ===
        lidar_readings = self._cast_rays([
            -math.pi/2, 
            -math.pi/4, 
            0, 
            math.pi/4, 
            math.pi/2
        ])
        
        # Store for rendering
        self.lidar_dist = lidar_readings
        
        # TOTAL SIZE: 
        # 0: Speed
        # 1: Local Vx (Forward)
        # 2: Local Vy (Lateral Drift)
        # 3: Angular Vel
        # 4: Prev Steer
        # 5: Target Dist
        # 6: Target Dir X (Forward)
        # 7: Target Dir Y (Right)
        # 8: On Track
        # 9-13: LIDAR (5)
        # = 14 Floats

        return np.array([
            speed_n,
            local_vx,
            local_vy,
            ang_vel,
            prev_steer,
            dist_n,
            tgt_dir_x,
            tgt_dir_y,
            on_track,
            *lidar_readings
        ], dtype=np.float32)

    def _cast_rays(self, angles: List[float]) -> List[float]:
        """
        Cast rays from car center to detect wall/track boundaries.
        Optimized using Numba JIT.
        """
        RAY_LENGTH = 150.0 # pixels
        
        # Prepare inputs for JIT
        px, py = self.car.pos
        h = self.car.heading
        
        # Convert angles to numpy array
        angles_np = np.array(angles, dtype=np.float32)
        
        # Call JIT function
        # This will be compiled on first run (might lag for 0.5s once)
        readings = cast_rays_jit(
            float(px), float(py), float(h),
            angles_np,
            float(RAY_LENGTH),
            self.track.occupancy_grid
        )
            
        return readings.tolist()

