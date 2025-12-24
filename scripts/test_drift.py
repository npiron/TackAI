#!/usr/bin/env python3
"""
Test de Dérapage - Vérifie le freinage et le grip
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import math
from src.game.track import build_track
from src.game.car import Car
from src.game.rendering import draw_track, draw_car
from src.utils.math_utils import vec
from src.core.constants import W, H, FPS, MAX_SPEED, ACCEL, BRAKE, FRICTION, TURN_SPEED

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("🏎️ Test de Dérapage - Freinage & Grip")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    track = build_track()
    
    car = Car(
        pos=vec(*track.start_pos),
        vel=vec(0, 0),
        heading=track.start_heading,
        speed=0.0,
        angular_velocity=0.0
    )
    
    print("🎮 Contrôles:")
    print("   ↑ - Accélérer")
    print("   ↓ - Freiner")
    print("   ← → - Tourner")
    print("   ESC - Quitter")
    print("\n💡 Astuce: Tourne + Freine = Dérapage !")
    
    running = True
    LATERAL_GRIP = 0.15  # Même valeur que dans env.py
    
    while running:
        # Input
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Actions
        accel = 1.0 if keys[pygame.K_UP] else 0.0
        brake = 1.0 if keys[pygame.K_DOWN] else 0.0
        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = 1.0
        
        # Physics (simplifié de env.py)
        # 1. Speed
        if accel > 0:
            car.speed += accel * ACCEL
        if brake > 0:
            car.speed -= brake * BRAKE
        car.speed *= FRICTION
        car.speed = max(0.0, min(car.speed, MAX_SPEED))
        
        # 2. Steering
        TURN_POWER = 0.05
        target_av = steer * TURN_SPEED
        car.angular_velocity += (target_av - car.angular_velocity) * TURN_POWER
        speed_factor = 1.0 - (car.speed / MAX_SPEED) * 0.3
        car.heading += car.angular_velocity * speed_factor
        
        # 3. Velocity (avec grip)
        intended_vx = math.cos(car.heading) * car.speed
        intended_vy = math.sin(car.heading) * car.speed
        intended_vel = vec(intended_vx, intended_vy)
        
        # GRIP - C'est ici que le dérapage se produit !
        car.vel = car.vel * (1.0 - LATERAL_GRIP) + intended_vel * LATERAL_GRIP
        
        # 4. Position
        car.pos += car.vel
        
        # Calculate drift angle
        drift_angle = 0.0
        if car.speed > 0.1:
            vel_angle = math.atan2(car.vel[1], car.vel[0])
            drift_angle = abs(car.heading - vel_angle)
            while drift_angle > math.pi:
                drift_angle -= 2 * math.pi
            drift_angle = abs(drift_angle)
        
        is_drifting = drift_angle > 0.26  # > 15 degrees
        
        # Draw
        draw_track(screen, track)
        draw_car(screen, car)
        
        # HUD
        speed_text = font.render(f"Vitesse: {car.speed:.1f} / {MAX_SPEED:.1f}", True, (255, 255, 255))
        screen.blit(speed_text, (10, 10))
        
        drift_text = font.render(f"Drift: {math.degrees(drift_angle):.1f}°", True, (255, 255, 0) if is_drifting else (255, 255, 255))
        screen.blit(drift_text, (10, 40))
        
        if is_drifting:
            status = font.render("🌀 DÉRAPAGE !", True, (255, 100, 100))
            screen.blit(status, (10, 70))
        
        if brake > 0:
            brake_text = font.render("🛑 FREIN", True, (255, 0, 0))
            screen.blit(brake_text, (10, 100))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
