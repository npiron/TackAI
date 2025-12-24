#!/usr/bin/env python3
"""
Watch Personal Best Replay
Rejoue la meilleure trajectoire enregistrée
"""

import sys
import os
import json
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game.track import build_track
from src.game.car import Car
from src.game.rendering import draw_track, draw_car, draw_ghost, fmt_time
from src.utils.math_utils import vec
from src.core.constants import W, H, FPS

def main():
    # Load replay
    if not os.path.exists("replays/personal_best.json"):
        print("❌ No Personal Best replay found!")
        print("   Train the AI first and wait for a PB to be saved.")
        return
    
    with open("replays/personal_best.json", "r") as f:
        replay = json.load(f)
    
    trajectory = replay["trajectory"]
    pb_time = replay["time"]
    
    print(f"🏆 Personal Best Replay")
    print(f"   Time: {pb_time:.2f}s")
    print(f"   Frames: {len(trajectory)}")
    print(f"\nPress SPACE to pause/resume")
    print(f"Press ESC to exit")
    print(f"Press R to restart\n")
    
    replay_seed = replay.get("track_seed", 42)
    
    # Setup
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"🏆 Personal Best Replay - {pb_time:.2f}s")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Rebuild the EXACT track used for the record
    print(f"   Track Seed: {replay_seed}")
    track = build_track(seed=replay_seed)
    
    # Replay state
    frame = 0
    paused = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    frame = 0
        
        # Update
        if not paused and frame < len(trajectory):
            frame += 1
        
        # Draw
        draw_track(screen, track)
        
        # Draw ghost trail
        if frame > 0:
            draw_ghost(screen, trajectory[:frame], color=(100, 200, 255))
        
        # Draw current car position
        if frame < len(trajectory):
            pos = trajectory[frame]
            # Calculate heading from trajectory
            heading = 0.0
            if frame > 0:
                prev_pos = trajectory[frame - 1]
                import math
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    heading = math.atan2(dy, dx)
            
            car = Car(
                pos=vec(pos[0], pos[1]),
                vel=vec(0, 0),
                heading=heading,
                speed=0.0
            )
            draw_car(screen, car)
        
        # HUD
        time_text = font.render(f"Time: {(frame / FPS):.2f}s / {pb_time:.2f}s", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))
        
        status = "PAUSED" if paused else "PLAYING"
        if frame >= len(trajectory):
            status = "FINISHED"
        status_text = font.render(status, True, (255, 255, 0))
        screen.blit(status_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
