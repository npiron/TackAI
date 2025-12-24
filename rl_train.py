import sys
import warnings
import time
sys.stdout.reconfigure(line_buffering=True)
print("🚀 Starting RL Training Script...")

import argparse
import os
import ast

# Lightweight imports only
print("⏳ Loading lightweight libraries...")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn

print("✅ Ready. Heavy libraries will load when training starts...")


def main():
    # Parse args first (fast)
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000, help="Total timesteps")
    parser.add_argument("--visual", action="store_true", help="Render training (slower, single core)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--run-id", type=str, default=str(int(time.time())), help="Unique run identifier")
    parser.add_argument("--load", type=str, default=None, help="Path to existing model to flatten curve (continue training)")
    parser.add_argument("--use-best-params", action="store_true", help="Load hyperparameters from logs/best_hyperparams.txt")
    parser.add_argument("--static", action="store_true", help="Disable procedural generation (use fixed track)")
    args = parser.parse_args()
    
    # NOW load the heavy stuff (only when actually training)
    print("⏳ Loading Pytorch (~1s)...")
    import torch
    
    print("⏳ Loading Stable Baselines3...")
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    
    print("⏳ Loading Game Components...")
    # Suppress Pygame's pkg_resources warning
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
    import pygame
    from src.rl.wrappers import GymTimeAttack
    
    print("✅ All imports loaded.")

    import multiprocessing

    # Device selection
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("⚡️ Apple Silicon GPU detected (MPS).")
    elif torch.cuda.is_available():
        device = "cuda"
        print("⚡️ NVIDIA GPU detected (CUDA).")
    else:
        print("🖥️  Using CPU.")
    
    # Environment setup
    procedural_mode = not args.static
    if procedural_mode:
        print("🔀 Procedural Generation ENABLED")
    else:
        print("⏹️ Procedural Generation DISABLED (Static Track)")

    if args.visual:
        # Single-core visual training
        n_cpu = 1
        print("🎨 Launching Visual training mode (single core)...")
        
        def make_env():
            env = GymTimeAttack(render_mode="human", procedural=procedural_mode)
            os.makedirs("data/monitoring", exist_ok=True)
            return Monitor(env, f"data/monitoring/monitor_visual_{args.run_id}.csv", info_keywords=("checkpoints_reached", "is_success", "lap_time"))
        
        venv = DummyVecEnv([make_env])
    else:
        # Headless Multi-core
        # DQN is Off-Policy, so parallel environments are less critical for sample efficiency than PPO,
        # but they speed up data collection.
        n_cpu = 16  # Fixed to 16 parallel environments for optimal training speed
        print(f"🚀 Launching Headless training on {n_cpu} parallel environments...")
        
        def make_env():
            env = GymTimeAttack(render_mode=None, procedural=procedural_mode)
            os.makedirs("data/monitoring", exist_ok=True)
            return Monitor(env, f"data/monitoring/monitor_headless_{args.run_id}_{os.getpid()}.csv", info_keywords=("checkpoints_reached", "is_success", "lap_time"))

        print("⏳ Creating environments...")
        # Avoid too many envs for DQN if RAM is valid concern, but 8-10 is fine.
        venv = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        print("✅ Environments ready.")

    # Wrap in VecNormalize (DQN can benefit from reward normalization too, but obs normalization is key)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 50000 // n_cpu),
        save_path="./data/checkpoints/",
        name_prefix=args.run_id,
        verbose=1
    )

    print(f"🧠 Initializing DQN model (Optimized for discrete control)...")
    print(f"   Buffer Size: 200,000")
    print(f"   Batch Size: 256")
    print(f"   Gradient Steps: 2 (improved)")
    print(f"   Exploration: 15% (improved)")
    print(f"   Architecture: [256, 256] with Dueling DQN")
    
    # Network Architecture
    policy_kwargs = dict(
        net_arch=[256, 256] 
        # n_quantiles? No, just standard DQN with Dueling (default in SB3)
    )
    
    # Learning Rate Schedule (decay over time to stabilize)
    def lr_schedule(progress_remaining):
        """
        Learning rate schedule for DQN:
        - Start: 1e-3 (exploration)
        - End: 1e-5 (fine-tuning)
        """
        return 1e-5 + (1e-3 - 1e-5) * progress_remaining

    # DQN Hyperparameters (Optimized for discrete action control)
    dqn_params = {
        "buffer_size": 200_000,   # Large replay buffer for stability
        "learning_starts": 5_000, # Fill buffer before training
        "batch_size": 256,
        "gamma": 0.99,            # Discount factor for discrete actions
        "train_freq": 4,          # Train every 4 env steps
        "gradient_steps": 2,      # More gradient steps per update (improved from 1)
        "target_update_interval": 1000, # Stabilize target network
        "exploration_fraction": 0.15,   # Explore 15% of total time (improved from 0.1)
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,  # Keep 5% random actions
        "max_grad_norm": 10,
        "learning_rate": lr_schedule,    # Progressive Learning Rate
    }

    if args.use_best_params and os.path.exists("data/optimization/best_hyperparams.txt"):
        print("🧪 Loading optimized hyperparameters from data/optimization/best_hyperparams.txt...")
        try:
            with open("data/optimization/best_hyperparams.txt", "r") as f:
                best_params = ast.literal_eval(f.read())
            
            # Update params with best_params
            
            # Extract architecture params first if present
            net_arch_type = best_params.pop("net_arch", None)
            
            if net_arch_type:
                architectures = {
                    "small": [64, 64],
                    "medium": [256, 256],
                    "large": [512, 512]
                }
                policy_kwargs["net_arch"] = architectures.get(net_arch_type, architectures["medium"])
                print(f"   Architecture: {net_arch_type} {policy_kwargs['net_arch']}")

            for k, v in best_params.items():
                if k in dqn_params:
                    dqn_params[k] = v
                else:
                    print(f"⚠️ Warning: Unknown param '{k}' in best_params. Ignoring.")
                    
            print(f"✅ Applied params: {dqn_params}")
        except Exception as e:
            print(f"⚠️ Failed to load best params: {e}")

    if args.load and os.path.exists(args.load):
        print(f"🔄 Loading existing model from {args.load}...")
        custom_objs = {
            "learning_rate": dqn_params["learning_rate"],
            "buffer_size": dqn_params["buffer_size"],
            "batch_size": dqn_params["batch_size"],
            "gamma": dqn_params["gamma"],
            "train_freq": dqn_params["train_freq"],
            "gradient_steps": dqn_params["gradient_steps"],
            "exploration_fraction": dqn_params["exploration_fraction"],
            "exploration_final_eps": dqn_params["exploration_final_eps"]
        }
        
        # Load the model but attach our new environment
        model = DQN.load(
            args.load,
            env=venv,
            device=device,
            print_system_info=True,
            custom_objects=custom_objs
        )
        print("✅ Model loaded successfully. Resuming training...")
    else:
        print(f"🧠 Initializing NEW DQN model...")
        model = DQN(
            "MlpPolicy",
            venv,
            device=device,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **dqn_params
        )

    print("🏃 Starting training (DQN)...")
    print("ℹ️  DQN gathers experience before training. Please wait for 'learning_starts' steps.")
    try:
        model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted manually.")
    
    print("💾 Saving model...")
    os.makedirs("data/models/production", exist_ok=True)
    model.save(f"data/checkpoints/{args.run_id}_final")
    venv.save(f"data/checkpoints/{args.run_id}_final_vecnormalize.pkl")
    print(f"✅ Saved to data/checkpoints/{args.run_id}_final.zip")
    print(f"💡 To use this model, copy it to data/models/production/")
    
    try:
        venv.close()
    except:
        pass


if __name__ == "__main__":
    # On macOS, 'spawn' is safer than 'fork' for libraries like Pygame/Torch
    # Reverting to default (which is usually spawn on macOS) to avoid deadlocks.
    # multiprocessing.set_start_method("spawn", force=True) 
    # Actually, we can just leave it to default.
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
