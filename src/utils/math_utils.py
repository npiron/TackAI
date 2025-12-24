
import math
import numpy as np
from typing import List, Tuple

def rotate_points(pts, angle: float, origin=(0, 0)):
    """Rotate a list of (x,y) points around origin by angle (radians)."""
    ox, oy = origin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    new_pts = []
    for (x, y) in pts:
        dx, dy = x - ox, y - oy
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        new_pts.append((ox + rx, oy + ry))
    return new_pts

def vec(x, y):
    return np.array([x, y], dtype=float)
