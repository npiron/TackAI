
import math
import numpy as np
import random
from typing import Optional, List, Tuple

from src.core.constants import *
from src.game.track import Track
from src.game.car import Car
from src.utils.math_utils import vec

class TimeAttackEnv:
    """
    Minimal env:
      obs = [x_norm, y_norm, vx_norm, vy_norm, heading_sin, heading_cos, speed_norm,
             cp_dx, cp_dy, fin_dx, fin_dy, on_track]
      action = (steer, accel, brake) in [-1..1], [0..1], [0..1]
    """
    def __init__(self, track: Track):
        self.track = track
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
        
        # 2. Speed-dependent steering (less grip at high speed)
        # At low speed: full turning power
        # At high speed: reduced turning power (more realistic)
        speed_factor = 1.0 - (self.car.speed / MAX_SPEED) * 0.3  # 30% reduction at max speed
        effective_turn_speed = TURN_SPEED * speed_factor
        
        # Update heading based on steering input
        self.car.heading += steer * effective_turn_speed
        
        # 3. Calculate intended velocity (where car WANTS to go)
        intended_vx = math.cos(self.car.heading) * self.car.speed
        intended_vy = math.sin(self.car.heading) * self.car.speed
        intended_vel = vec(intended_vx, intended_vy)
        
        # 4. Apply lateral friction (prevent sliding/drifting)
        # The car's velocity gradually aligns with its heading direction
        # This creates grip and prevents unrealistic serpentine movement
        
        # Lateral friction strength (0.0 = ice, 1.0 = instant grip)
        LATERAL_GRIP = 0.15  # Lower = more drift, Higher = more grip
        
        # Gradually transition current velocity towards intended velocity
        self.car.vel = self.car.vel * (1.0 - LATERAL_GRIP) + intended_vel * LATERAL_GRIP
        
        # 5. Update Position
        self.car.pos += self.car.vel

        # 4a. Check wall collision (screen boundaries)
        if self.track.hits_wall(self.car.pos):
            # Hard bounce - car hit the invisible wall
            self.car.speed *= 0.3  # Severe penalty
            # Clamp position to stay in bounds
            self.car.pos[0] = max(5, min(W - 5, self.car.pos[0]))
            self.car.pos[1] = max(5, min(H - 5, self.car.pos[1]))

        # 4b. Check track collision (grass vs road)
        if not self.track.inside_track(self.car.pos):
            # Off-road penalty (less severe than wall)
            self.car.speed *= COLLISION_SPEED_LOSS
            self.off_track_counter += 1
            
            # If off-track for too long, terminate episode (stuck)
            if self.off_track_counter > 100:  # ~2 seconds at 50 FPS
                self.done = True
        else:
            self.off_track_counter = 0  # Reset when back on track

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
        
        fx, fy = self.track.finish_pos
        dist = np.linalg.norm(self.car.pos - vec(fx, fy))
        return dist < FINISH_RADIUS

    def _obs(self):
        # Normalize inputs for Neural Net
        x, y = self.car.pos
        vx, vy = self.car.vel
        
        # Normalized Pos (0..1)
        xn = x / W
        yn = y / H
        
        # Velocity matches heading/speed, but let's give raw components
        vxn = vx / MAX_SPEED
        vyn = vy / MAX_SPEED
        
        # Heading trig
        h_sin = math.sin(self.car.heading)
        h_cos = math.cos(self.car.heading)
        
        speed_n = self.car.speed / MAX_SPEED
        
        # Relative vector to NEXT Checkpoint (or Finish)
        next_obj = None
        for i, visited in enumerate(self.cp_visited):
            if not visited:
                next_obj = self.track.checkpoints[i][:2] # (x, y)
                break
        
        if next_obj is None:
            # All CPs done, target is finish
            next_obj = self.track.finish_pos
            
        dx = (next_obj[0] - x) / W
        dy = (next_obj[1] - y) / H
        
        # Vector to finish (always useful?)
        fx, fy = self.track.finish_pos
        fdx = (fx - x) / W
        fdy = (fy - y) / H
        
        on_track = 1.0 if self.track.inside_track(self.car.pos) else 0.0

        return np.array([
            xn, yn, 
            vxn, vyn, 
            h_sin, h_cos,
            speed_n,
            dx, dy,
            fdx, fdy,
            on_track
        ], dtype=np.float32)
