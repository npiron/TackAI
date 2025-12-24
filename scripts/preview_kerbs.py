#!/usr/bin/env python3
"""
Quick visual test to see the new kerbs (vibreurs) on the track.
Run this to preview the track with kerbs before training.
"""

import pygame
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.track import build_track
from src.game.rendering import draw_track
from src.game.car import Car
from src.core.constants import W, H
from src.utils.math_utils import vec

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Track Preview - Kerbs Test")
    clock = pygame.time.Clock()
    
    # Build track
    track = build_track()
    
    # Create a test car
    car = Car(
        pos=vec(*track.start_pos),
        vel=vec(0, 0),
        heading=track.start_heading,
        speed=0.0
    )
    
    print("🏁 Track Preview - Simple Rectangle")
    print(f"✅ {len(track.checkpoints)} checkpoints")
    print("\nPress ESC to exit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Draw
        draw_track(screen, track)
        
        # Draw car at start
        from src.game.rendering import draw_car
        draw_car(screen, car)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
