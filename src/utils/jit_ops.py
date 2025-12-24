
import numpy as np
from numba import jit

@jit(nopython=True)
def cast_rays_jit(pos_x, pos_y, heading, angles, ray_length, grid):
    """
    Fast Raycasting using Numba.
    grid: 2D numpy array (bool or int), where True/1 is WALKABLE (Road), False/0 is WALL.
    """
    height, width = grid.shape
    n_rays = len(angles)
    readings = np.empty(n_rays, dtype=np.float32)
    
    step_size = 2.0 # Haute précision grâce à Numba !
    
    for i in range(n_rays):
        rel_angle = angles[i]
        abs_angle = heading + rel_angle
        
        rx = np.cos(abs_angle)
        ry = np.sin(abs_angle)
        
        found_wall = False
        final_dist = ray_length
        
        # Manual loop for ray steps
        # range(start, stop, step)
        # On commence un peu loin (4px) pour éviter de taper sa propre carrosserie
        n_steps = int((ray_length - 4.0) / step_size)
        
        for s in range(n_steps):
            d = 4.0 + s * step_size
            
            px = int(pos_x + rx * d)
            py = int(pos_y + ry * d)
            
            # Check bounds (hit screen edge = wall)
            if px < 0 or px >= width or py < 0 or py >= height:
                final_dist = d
                found_wall = True
                break
            
            # Check collision
            # grid[y, x] == 0 means NOT Road -> Wall
            if not grid[py, px]:
                final_dist = d
                found_wall = True
                break
        
        readings[i] = final_dist / ray_length
        
    return readings
