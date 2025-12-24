
import pygame
import math
from typing import List, Tuple
from src.core.constants import *
from src.game.track import Track
from src.game.car import Car
from src.utils.math_utils import vec

def draw_track(screen, track: Track):
    # Procédural (Surface pré-rendue)
    if hasattr(track, 'visual_surface') and track.visual_surface is not None:
        screen.blit(track.visual_surface, (0, 0))
    else:
        # Fallback Legacy (Rectangles)
        screen.fill(COLOR_GRASS)
        if track.rect_outer:
            pygame.draw.rect(screen, COLOR_ROAD, track.rect_outer, border_radius=20)
            pygame.draw.rect(screen, COLOR_BORDER, track.rect_outer, width=4, border_radius=20)
            pygame.draw.rect(screen, COLOR_BORDER, track.rect_inner, width=4, border_radius=20)
            pygame.draw.rect(screen, COLOR_GRASS, track.rect_inner, border_radius=20)

    # Checkpoints
    for cx, cy, angle_or_orient in track.checkpoints:
        half_w = 50
        
        # Gestion hybride: angle (float) ou orientation character (str)
        if isinstance(angle_or_orient, str):
            # Legacy 'V'/'H'
            if angle_or_orient == 'V':
                angle = math.pi / 2
            else:
                angle = 0
        else:
            angle = angle_or_orient

        # Calcul des points extrémités de la ligne de checkpoint
        # Perpendiculaire à la route (l'angle stocké est déjà supposé être celui de la ligne de CP)
        c = math.cos(angle)
        s = math.sin(angle)
        
        p1 = (cx - c * half_w, cy - s * half_w)
        p2 = (cx + c * half_w, cy + s * half_w)
        
        pygame.draw.line(screen, COLOR_CHECKPOINT, p1, p2, width=8)
    
    # Finish
    # Finish
    if track.finish_pos:
        if len(track.finish_pos) == 3:
            fx, fy, f_angle = track.finish_pos
            half_w = 50
            c = math.cos(f_angle)
            s = math.sin(f_angle)
            p1 = (fx - c * half_w, fy - s * half_w)
            p2 = (fx + c * half_w, fy + s * half_w)
            # Draw Checkered Line (White Thick)
            pygame.draw.line(screen, (255, 255, 255), p1, p2, width=12)
            # Maybe a black stripe in middle for style
            pygame.draw.line(screen, (0, 0, 0), p1, p2, width=4)
        else:
            fx, fy = track.finish_pos
            pygame.draw.circle(screen, (255, 255, 255), (int(fx), int(fy)), 12)
    
    # Start
    if track.start_pos:
        sx, sy = track.start_pos
        pygame.draw.circle(screen, (255, 255, 0), (int(sx), int(sy)), 8)


def draw_ghost(screen, pts: List[Tuple[float, float]], color=(110, 110, 140)):
    if len(pts) < 2:
        return
    pygame.draw.lines(screen, color, False, [(int(x), int(y)) for x, y in pts], width=2)


def draw_car(screen, car: Car):
    p = car.pos
    ang = car.heading
    f = vec(math.cos(ang), math.sin(ang))
    r = vec(-math.sin(ang), math.cos(ang))

    nose = p + f * 18
    left = p - f * 12 - r * 10
    right = p - f * 12 + r * 10

    pygame.draw.polygon(
        screen,
        (230, 50, 50),
        [(int(nose[0]), int(nose[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))],
    )

def draw_lidar(screen, car: Car, readings: List[float]):
    """Draw LIDAR rays."""
    if not readings:
        return
        
    RAY_LENGTH = 150.0
    angles = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]
    
    cx, cy = car.pos
    
    for i, dist_norm in enumerate(readings):
        rel_angle = angles[i]
        abs_angle = car.heading + rel_angle
        
        val = max(0.0, min(1.0, dist_norm))
        color = (int(255 * (1-val)), int(255 * val), 0)
        
        rx = math.cos(abs_angle)
        ry = math.sin(abs_angle)
        
        end_x = cx + rx * (dist_norm * RAY_LENGTH)
        end_y = cy + ry * (dist_norm * RAY_LENGTH)
        
        pygame.draw.line(screen, color, (cx, cy), (end_x, end_y), 1)
        pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

def fmt_time(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:06.3f}"
