import sys
sys.stdout.reconfigure(line_buffering=True)
print("🤖 AI PLAY STARTING...")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import os

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.wrappers import GymTimeAttack
from src.core.constants import FPS

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to DQN model .zip")
    parser.add_argument("--env", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--no-procedural", action="store_true", help="Disable procedural track generation (use fixed track)")
    args = parser.parse_args()

    # Default paths to check (in priority order)
    model_path = args.model
    if not model_path:
        # Check production models first
        if os.path.exists("data/models/production/best_model.zip"): 
            model_path = "data/models/production/best_model.zip"
        # Then checkpoints
        elif os.path.exists("data/checkpoints/latest_model.zip"): 
            model_path = "data/checkpoints/latest_model.zip"
        # Legacy locations
        elif os.path.exists("dqn_timeattack.zip"): 
            model_path = "dqn_timeattack.zip"
            
    # If still not found, try to find ANY zip in checkpoints
    if not model_path and os.path.exists("data/checkpoints"):
        files = [f for f in os.listdir("data/checkpoints") if f.endswith(".zip")]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join("data/checkpoints", x)), reverse=True)
            model_path = os.path.join("data/checkpoints", files[0])

    env_path = args.env
    
    # 1. Try to find corresponding pkl for the specific model
    if not env_path and model_path:
        base_name = model_path.replace(".zip", "")
        # Try local sibling
        candidate = base_name + "_vecnormalize.pkl"
        if os.path.exists(candidate):
            env_path = candidate
    
    # 2. Try generic checkpoint
    if not env_path:
         if os.path.exists("data/checkpoints/vecnormalize.pkl"): 
            env_path = "data/checkpoints/vecnormalize.pkl"
            
    # 3. Fallback to production
    if not env_path:
        if os.path.exists("data/models/production/vecnormalize.pkl"): 
            env_path = "data/models/production/vecnormalize.pkl"

    def make_env():
        # IMPORTANT: render_mode='human' to display the game window
        # Ensure procedural generation is enabled (or match training config)
        return GymTimeAttack(render_mode="human", procedural=not args.no_procedural)

    if not model_path or not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        print(f"   Checked: {model_path if model_path else 'data/checkpoints/*.zip'}")
        return

    if not env_path or not os.path.exists(env_path):
        print("⚠️ Warning: VecNormalize file not found. Running without normalization.")

    print(f"🔄 Loading Checkpoint:\n   Model: {model_path}\n   Env:   {env_path}")

    venv = DummyVecEnv([make_env])
    if env_path and os.path.exists(env_path):
        try:
            venv = VecNormalize.load(env_path, venv)
            venv.training = False
            venv.norm_reward = False
            print("✅ VecNormalize loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load VecNormalize: {e}")
            print("   -> Running WITHOUT normalization.")
    else:
        print("⚠️ Running without VecNormalize wrapper.")

    print("⏳ Loading DQN Model...")
    model = DQN.load(model_path, env=venv)
    print("✅ Model loaded! Starting simulation...")
    print("🎮 Controls:")
    print("   ESC - Quit")
    print("   R - Reset episode")
    print("   SPACE - Pause/Resume")
    
    # Get the underlying env for HUD access
    base_env = venv.envs[0].unwrapped
    
    # Setup clock for FPS control
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    
    obs = venv.reset()
    episode = 0
    paused = False
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("👋 Goodbye!")
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("👋 Goodbye!")
                    return
                if event.key == pygame.K_r:
                    print("🔄 Resetting episode...")
                    obs = venv.reset()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("⏸️ PAUSED" if paused else "▶️ RESUMED")
        
        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            
            if done:
                episode += 1
                checkpoints = info[0].get('checkpoints_reached', 0)
                success = info[0].get('is_success', False)
                lap_time = info[0].get('lap_time', 0)
                
                status = "✅ FINISHED" if success else "❌ FAILED"
                print(f"Episode #{episode} {status} | Checkpoints: {checkpoints}/9 | Time: {lap_time:.2f}s | Reward: {reward[0]:.2f}")
                obs = venv.reset()
        
        # Draw HUD overlay
        if hasattr(base_env, 'ta') and hasattr(base_env.ta, 'screen'):
            screen = base_env.ta.screen
            
            # Timer
            time_text = font.render(f"Time: {base_env.ta.t:.2f}s", True, (255, 255, 255))
            screen.blit(time_text, (10, 10))
            
            # Checkpoints
            cp_count = sum(base_env.ta.cp_visited)
            cp_text = font.render(f"CP: {cp_count}/{len(base_env.ta.track.checkpoints)}", True, (255, 255, 255))
            screen.blit(cp_text, (10, 45))
            
            # Episode
            ep_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
            screen.blit(ep_text, (10, 80))
            
            # Status
            if paused:
                pause_text = font.render("PAUSED", True, (255, 255, 0))
                screen.blit(pause_text, (10, 115))
            
            pygame.display.flip()
        
        # Control FPS
        clock.tick(FPS)


if __name__ == "__main__":
    main()
