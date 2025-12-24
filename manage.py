#!/usr/bin/env python3
"""
Trackmania RL Clone - Management Script
========================================
Unified command-line interface for all project operations.

Usage:
    python manage.py <command> [options]

Commands:
    play            - Play the game manually
    train           - Train the RL agent
    watch           - Watch the trained AI play
    optimize        - Run hyperparameter optimization
    dashboard       - Launch the web dashboard
    replay          - Watch personal best replay
    info            - Display project information
    clean           - Clean cache files

Examples:
    python manage.py play
    python manage.py train --steps 1000000 --visual
    python manage.py watch --model logs/best_model.zip
    python manage.py optimize --trials 50
    python manage.py dashboard
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

# AUTO-VENV DETECTION
# If we are not in a venv, but a .venv directory exists, re-exec this script using the venv's python.
def ensure_venv():
    # Check if running inside a virtual environment
    is_venv = (sys.prefix != sys.base_prefix)
    
    # Path to local .venv
    venv_path = Path(__file__).parent / ".venv"
    venv_python = venv_path / "bin" / "python"
    
    if not is_venv and venv_path.exists() and venv_python.exists():
        # Re-execute the script with the venv python
        # print(f"🔄 Switching to virtual environment: {venv_python}")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Run auto-detection immediately
ensure_venv()


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def run_command(cmd, description=None):
    """Run a shell command and handle errors"""
    if description:
        print_info(description)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def cmd_play(args):
    """Launch the game for manual play"""
    print_header("🎮 MANUAL PLAY MODE")
    print_info("Controls:")
    print("  • Arrow Keys: Steer and accelerate")
    print("  • R: Reset")
    print("  • Space: Pause")
    print("  • G: Toggle ghost")
    print("  • Esc: Quit\n")
    
    return run_command([sys.executable, "trackmania_clone.py"])


def cmd_train(args):
    """Train the RL agent"""
    print_header("🚀 RL TRAINING")
    
    cmd = [sys.executable, "rl_train.py"]
    
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    if args.visual:
        cmd.append("--visual")
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.load:
        cmd.extend(["--load", args.load])
    if args.use_best_params:
        cmd.append("--use-best-params")
    if args.static:
        cmd.append("--static")
    
    return run_command(cmd, f"Starting training with {args.steps or 2000000} steps...")


def cmd_watch(args):
    """Watch the trained AI play"""
    print_header("🤖 AI PLAY MODE")
    
    cmd = [sys.executable, "rl_play.py"]
    
    if args.model:
        cmd.extend(["--model", args.model])
    if args.env:
        cmd.extend(["--env", args.env])
    
    return run_command(cmd, "Loading AI model...")


def cmd_optimize(args):
    """Run hyperparameter optimization"""
    print_header("🧪 HYPERPARAMETER OPTIMIZATION")
    
    cmd = [sys.executable, "rl_optimize.py"]
    
    if args.trials:
        cmd.extend(["--trials", str(args.trials)])
    if args.timeout:
        cmd.extend(["--timeout", str(args.timeout)])
    
    return run_command(cmd, f"Starting optimization with {args.trials or 100} trials...")


def cmd_dashboard(args):
    """Launch the web dashboard"""
    print_header("📊 WEB DASHBOARD")
    
    cmd = [sys.executable, "-m", "dashboard.main"]
    
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.share:
        cmd.append("--share")
    
    print_info("Dashboard will be available at http://localhost:7860")
    return run_command(cmd)


def cmd_replay(args):
    """Watch personal best replay"""
    print_header("🎬 REPLAY MODE")
    
    cmd = [sys.executable, "watch_pb_replay.py"]
    
    if args.file:
        cmd.extend(["--file", args.file])
    
    return run_command(cmd, "Loading replay...")


def cmd_clean(args):
    """Clean cache and temporary files"""
    print_header("🧹 CLEANING PROJECT")
    
    patterns = [
        "__pycache__",
        ".pytest_cache",
        ".DS_Store"
    ]
    
    if args.logs:
        print_warning("This will delete all logs and checkpoints!")
        response = input("Are you sure? (y/N): ")
        if response.lower() == 'y':
            patterns.extend(["logs/*.csv", "logs/*.zip", "logs/*.pkl"])
    
    count = 0
    for pattern in patterns:
        if '*' in pattern:
            import glob
            for path in glob.glob(f"**/{pattern}", recursive=True):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        count += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        count += 1
                except Exception as e:
                    print_warning(f"Could not delete {path}: {e}")
        else:
            for root, dirs, files in os.walk('.'):
                if pattern in dirs:
                    path = os.path.join(root, pattern)
                    try:
                        shutil.rmtree(path)
                        count += 1
                        print_info(f"Removed: {path}")
                    except Exception as e:
                        print_warning(f"Could not delete {path}: {e}")
    
    print_success(f"Cleaned {count} items")
    return True


def cmd_info(args):
    """Display project information"""
    print_header("📋 PROJECT INFORMATION")
    
    # Project structure
    print(f"{Colors.BOLD}Project Structure:{Colors.ENDC}")
    print("  • src/          - Source code")
    print("  • dashboard/    - Web dashboard")
    print("  • logs/         - Training logs and checkpoints")
    print("  • replays/      - Saved replays")
    print("  • tests/        - Test suite")
    
    # Check for models
    print(f"\n{Colors.BOLD}Available Models:{Colors.ENDC}")
    model_paths = [
        "ppo_timeattack.zip",
        "logs/ppo_timeattack.zip",
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            found_models.append(f"  ✓ {path} ({size:.2f} MB)")
    
    if found_models:
        print("\n".join(found_models))
    else:
        print("  ⚠️  No trained models found")
    
    # Check for logs
    print(f"\n{Colors.BOLD}Training Logs:{Colors.ENDC}")
    if os.path.exists("logs"):
        log_files = list(Path("logs").glob("*.csv"))
        checkpoint_files = list(Path("logs").glob("*.zip"))
        print(f"  • CSV logs: {len(log_files)}")
        print(f"  • Checkpoints: {len(checkpoint_files)}")
    else:
        print("  ⚠️  No logs directory found")
    
    # Check for replays
    print(f"\n{Colors.BOLD}Replays:{Colors.ENDC}")
    if os.path.exists("replays"):
        replay_files = list(Path("replays").glob("*.json"))
        print(f"  • Saved replays: {len(replay_files)}")
    else:
        print("  ⚠️  No replays directory found")
    
    # Dependencies
    print(f"\n{Colors.BOLD}Environment:{Colors.ENDC}")
    print(f"  • Python: {sys.version.split()[0]}")
    print(f"  • Virtual env: {'✓ Active' if sys.prefix != sys.base_prefix else '✗ Not active'}")
    
    try:
        import pygame
        print(f"  • Pygame: {pygame.version.ver}")
    except ImportError:
        print("  • Pygame: ✗ Not installed")
    
    try:
        import stable_baselines3
        print(f"  • Stable-Baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print("  • Stable-Baselines3: ✗ Not installed")
    
    try:
        import torch
        print(f"  • PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("  • Device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            print("  • Device: CUDA")
        else:
            print("  • Device: CPU")
    except ImportError:
        print("  • PyTorch: ✗ Not installed")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Trackmania RL Clone - Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Play command
    subparsers.add_parser('play', help='Play the game manually')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--steps', type=int, help='Total training steps')
    train_parser.add_argument('--visual', action='store_true', help='Visual training mode')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--run-id', type=str, help='Run identifier')
    train_parser.add_argument('--load', type=str, help='Load existing model')
    train_parser.add_argument('--use-best-params', action='store_true', help='Use optimized hyperparameters')
    train_parser.add_argument('--static', action='store_true', help='Disable procedural generation (use fixed track)')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch trained AI play')
    watch_parser.add_argument('--model', type=str, help='Path to model file')
    watch_parser.add_argument('--env', type=str, help='Path to VecNormalize file')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run hyperparameter optimization')
    optimize_parser.add_argument('--trials', type=int, help='Number of trials')
    optimize_parser.add_argument('--timeout', type=int, help='Timeout per trial (seconds)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dashboard_parser.add_argument('--port', type=int, help='Port number')
    dashboard_parser.add_argument('--share', action='store_true', help='Create public share link')
    
    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Watch personal best replay')
    replay_parser.add_argument('--file', type=str, help='Replay file path')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cache and temporary files')
    clean_parser.add_argument('--logs', action='store_true', help='Also clean logs and checkpoints')
    
    # Info command
    subparsers.add_parser('info', help='Display project information')

    # Setup command
    subparsers.add_parser('setup', help='Setup and verify environment')
    
    args = parser.parse_args()
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 0
    
    def cmd_setup(args):
        """Setup and verify environment"""
        print_header("⚙️  ENVIRONMENT SETUP")
        
        # Check Python version
        print_info(f"Python version: {sys.version.split()[0]}")
        
        # Check virtual environment (handled by auto-detect now, but good to verify)
        if sys.prefix == sys.base_prefix:
            print_warning("Virtual environment NOT active (Auto-detect failed?)")
        else:
            print_success(f"Virtual environment active: {sys.prefix}")
        
        # Check dependencies
        print_info("Checking dependencies...")
        try:
            import stable_baselines3
            import optuna
            print_success("Dependencies found")
        except ImportError as e:
            print_warning(f"Missing dependency: {e.name}")
            response = input("\nInstall dependencies? (y/N): ")
            if response.lower() == 'y':
                return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "Installing...")
        
        # Create directories
        for d in ["data", "logs", "replays"]:
            os.makedirs(d, exist_ok=True)
            
        print_success("\n✨ Setup complete!")
        return True

    # Command dispatch
    commands = {
        'play': cmd_play,
        'train': cmd_train,
        'watch': cmd_watch,
        'optimize': cmd_optimize,
        'dashboard': cmd_dashboard,
        'replay': cmd_replay,
        'clean': cmd_clean,
        'info': cmd_info,
        'setup': cmd_setup,
    }
    
    handler = commands.get(args.command)
    if handler:
        try:
            success = handler(args)
            return 0 if success else 1
        except KeyboardInterrupt:
            print_warning("\n\nInterrupted by user")
            return 130
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print_error(f"Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
