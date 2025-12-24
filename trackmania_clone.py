import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import time
import pygame

t0 = time.time()

# -----------------------------
# Config
# -----------------------------
W, H = 1100, 700
FPS = 60
import math
from typing import List, Tuple

from src.core.constants import *
from src.game.track import build_track
from src.game.env import TimeAttackEnv
from src.game.rendering import draw_track, draw_car, draw_ghost, fmt_time

print(f"⏱️ Imports loaded in {time.time()-t0:.3f}s")

def run_game():
    print("🎮 TRACKMANIA CLONE STARTING...")
    pygame.init()
    print("✅ Pygame Initialized. Creating Window...")
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Trackmania Clone (Top-down Time Attack) - Pygame")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", FONT_SIZE)

    track = build_track()
    env = TimeAttackEnv(track)

    running = True
    paused = False

    help_lines = [
        "Controls: ←/→ steer | ↑ accel | ↓ brake | R reset | SPACE pause | G toggle ghost | ESC quit",
        "Goal: visit all checkpoints then reach finish. Beat your best time (ghost).",
    ]
    show_ghost = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_g:
                    show_ghost = not show_ghost

        keys = pygame.key.get_pressed()
        steer = (-1.0 if keys[pygame.K_LEFT] else 0.0) + (1.0 if keys[pygame.K_RIGHT] else 0.0)
        throttle = 1.0 if keys[pygame.K_UP] else 0.0
        brake = 1.0 if keys[pygame.K_DOWN] else 0.0

        if not paused:
            env.step((steer, throttle, brake))

        # Rendering Logic
        draw_track(screen, track)

        if show_ghost and env.best_time is not None:
            draw_ghost(screen, env.best_ghost, color=(70, 160, 220))
        draw_ghost(screen, env.ghost[-400:], color=(90, 90, 120))

        draw_car(screen, env.car)

        # checkpoints labels
        for i, (cx, cy, orient) in enumerate(track.checkpoints):
            label = font.render(str(i + 1), True, (220, 220, 240))
            screen.blit(label, (int(cx) - 6, int(cy) - 10))
            if env.cp_visited[i]:
                # Draw a small indicator circle on the checkpoint center
                pygame.draw.circle(screen, (70, 255, 120), (int(cx), int(cy)), 6)

        cur = font.render(f"Time: {fmt_time(env.t)}", True, (235, 235, 245))
        screen.blit(cur, (14, 14))

        best = "--:--.---" if env.best_time is None else fmt_time(env.best_time)
        best_surf = font.render(f"Best: {best}", True, (210, 210, 235))
        screen.blit(best_surf, (14, 40))

        status = ""
        if env.done:
            if all(env.cp_visited):
                status = "🏆 FINISHED! (R to restart)"
            else:
                status = "❌ FAILED (Off-Track Timeout) - R to restart"
        elif paused:
            status = "⏸️ PAUSED"

        if status:
            color = (120, 255, 120) if "FINISHED" in status else (255, 100, 100)
            if "PAUSED" in status: color = (240, 210, 140)
            st = font.render(status, True, color)
            screen.blit(st, (14, 66))

        y = H - 48
        for line in help_lines:
            surf = font.render(line, True, (240, 240, 250))
            screen.blit(surf, (14, y))
            y += 22

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_game()
