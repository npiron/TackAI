"""
Track geometry utilities for creating smooth racing circuits.
Uses polygons and Bezier curves instead of rectangles.
"""

import math
import numpy as np
from typing import List, Tuple
import pygame


def bezier_curve(p0, p1, p2, p3, num_points=20):
    """
    Generate points along a cubic Bezier curve.
    p0, p1, p2, p3 are control points (x, y tuples)
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        # Cubic Bezier formula
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def circular_arc(center, radius, start_angle, end_angle, num_points=15):
    """
    Generate points along a circular arc.
    """
    points = []
    angle_range = end_angle - start_angle
    for i in range(num_points + 1):
        angle = start_angle + (i / num_points) * angle_range
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points


def offset_polygon(points, offset):
    """
    Create an offset polygon (for track width).
    Positive offset = outward, negative = inward.
    Uses simple perpendicular offset at each point.
    """
    if len(points) < 3:
        return points
        
    offset_points = []
    n = len(points)
    
    for i in range(n):
        # Get three consecutive points
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        
        # Vector from prev to curr
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        
        # Vector from curr to next  
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            offset_points.append(p_curr)
            continue
            
        # Normalize
        v1 = (v1[0]/len1, v1[1]/len1)
        v2 = (v2[0]/len2, v2[1]/len2)
        
        # Perpendiculars (rotate 90° counterclockwise)
        n1 = (-v1[1], v1[0])
        n2 = (-v2[1], v2[0])
        
        # Average the normals
        avg_normal = ((n1[0] + n2[0])/2, (n1[1] + n2[1])/2)
        len_normal = math.sqrt(avg_normal[0]**2 + avg_normal[1]**2)
        
        if len_normal < 0.001:
            # Degenerate case - use first normal
            avg_normal = n1
            len_normal = 1.0
        else:
            avg_normal = (avg_normal[0]/len_normal, avg_normal[1]/len_normal)
        
        # Calculate scale factor to maintain offset distance
        # (compensate for angle between edges)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        angle = math.acos(max(-1.0, min(1.0, dot)))
        
        if abs(angle) < 0.01:  # Nearly straight
            scale = 1.0
        else:
            scale = 1.0 / math.sin(angle / 2 + math.pi/2)
            scale = max(0.5, min(3.0, abs(scale)))  # Clamp to avoid extreme values
        
        # Apply offset
        offset_point = (
            p_curr[0] + avg_normal[0] * offset * scale,
            p_curr[1] + avg_normal[1] * offset * scale
        )
        offset_points.append(offset_point)
    
    return offset_points


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
