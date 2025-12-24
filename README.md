# Trackmania RL Clone 🏎️

A **Trackmania-inspired top-down time attack** game made with **Pygame**, featuring a **Reinforcement Learning agent (PPO)** trained with **Stable-Baselines3**.

## ✨ Features

- ✅ Runs on macOS (Apple Silicon optimized with MPS)
- ✅ Human-playable with smooth controls
- ✅ RL training with PPO algorithm
- ✅ Best-time ghost replay system
- ✅ Web dashboard for monitoring training
- ✅ Hyperparameter optimization with Optuna

## 🚀 Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Using the Management Script

The project includes a unified management script for all operations:

```bash
# View all available commands
python3 manage.py --help

# Play the game manually
python3 manage.py play

# Train the AI agent
python3 manage.py train

# Watch the trained AI play
python3 manage.py watch

# Launch the web dashboard
python3 manage.py dashboard

# View project information
python3 manage.py info
```

## 🎮 Manual Play

```bash
python3 manage.py play
# or
python3 trackmania_clone.py
```

**Controls:**
- **Arrow Keys**: Steer and accelerate
- **R**: Reset
- **Space**: Pause
- **G**: Toggle best ghost
- **Esc**: Quit

**Goal:** Hit all checkpoints and finish the lap to beat your best time!

## 🤖 AI Training

### Basic Training

```bash
python3 manage.py train
# or
python3 rl_train.py
```

### Advanced Training Options

```bash
# Visual training mode (slower, single core)
python3 manage.py train --visual

# Custom number of steps
python3 manage.py train --steps 1000000

# Use optimized hyperparameters
python3 manage.py train --use-best-params

# Continue training from checkpoint
python3 manage.py train --load logs/checkpoint.zip
```

**Outputs:**
- `ppo_timeattack.zip` - Trained model
- `vecnormalize.pkl` - Normalization statistics
- `logs/` - Training logs and checkpoints

## 👀 Watch AI Play

```bash
python3 manage.py watch
# or
python3 rl_play.py
```

**Controls while watching:**
- **Space**: Pause/Resume
- **R**: Reset episode
- **Esc**: Quit

## 📊 Web Dashboard

Launch the interactive web dashboard to monitor training:

```bash
python3 manage.py dashboard
```

The dashboard provides:
- Real-time training metrics
- Learning curves and analytics
- Process management
- System statistics

## 🔧 Hyperparameter Optimization

Optimize training hyperparameters using Optuna:

```bash
python3 manage.py optimize --trials 50
# or
python3 rl_optimize.py --trials 50
```

## 📁 Project Structure

```
.
├── README.md              # This file
├── LICENSE                # License information
├── requirements.txt       # Python dependencies
├── manage.py             # Unified management script
├── trackmania_clone.py   # Manual play script
├── rl_train.py           # Training script
├── rl_play.py            # AI play script
├── rl_optimize.py        # Hyperparameter optimization
├── watch_pb_replay.py    # Replay viewer
├── rewards_config.json   # Reward configuration
├── pytest.ini            # Test configuration
├── src/                  # Source code
│   ├── core/            # Core game logic
│   ├── game/            # Game components
│   ├── rl/              # RL wrappers and utilities
│   └── ...
├── dashboard/            # Web dashboard
├── docs/                 # Documentation
│   ├── HYPERPARAMETERS_GUIDE.md
│   ├── OPTIMIZATION_GUIDE.md
│   ├── REPLAY_GUIDE.md
│   ├── GAMEPLAY_MECHANICS.md
│   ├── ANTI_REGRESSION_GUIDE.md
│   └── HYPERPARAMS_EDITOR.md
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── data/                 # Generated data (gitignored)
│   ├── models/          # Trained models
│   │   ├── production/  # Production-ready models
│   │   └── archive/     # Archived models
│   ├── checkpoints/     # Training checkpoints
│   ├── monitoring/      # Training metrics (CSV)
│   ├── logs/            # Application logs
│   │   ├── training/    # Training logs
│   │   ├── game/        # Game logs
│   │   └── ai/          # AI play logs
│   └── optimization/    # Hyperparameter optimization results
└── replays/              # Saved replays
```

## 📚 Documentation

For detailed information, see the documentation in the `docs/` folder:

- **[Hyperparameters Guide](docs/HYPERPARAMETERS_GUIDE.md)** - Understanding and tuning hyperparameters
- **[Optimization Guide](docs/OPTIMIZATION_GUIDE.md)** - Hyperparameter optimization strategies
- **[Replay Guide](docs/REPLAY_GUIDE.md)** - Using the replay system
- **[Gameplay Mechanics](docs/GAMEPLAY_MECHANICS.md)** - Game mechanics and physics
- **[Anti-Regression Guide](docs/ANTI_REGRESSION_GUIDE.md)** - Preventing training regression
- **[Hyperparams Editor](docs/HYPERPARAMS_EDITOR.md)** - Editing hyperparameters

## 🧪 Testing

Run the test suite:

```bash
pytest
# or
python3 -m pytest -v
```

## 🧹 Maintenance

Clean cache and temporary files:

```bash
python3 manage.py clean

# Also clean logs and checkpoints (careful!)
python3 manage.py clean --logs
```

## 🎯 Reward Shaping

The RL agent uses **reward shaping** for faster learning:
- Distance to next checkpoint
- Speed bonus
- Off-track penalty
- Checkpoint completion rewards

Configuration can be modified in `rewards_config.json`.

## 💡 Future Improvements

- [ ] Add raycast sensors for smoother wall avoidance
- [ ] Multiple tracks to reduce overfitting
- [ ] Procedurally generated tracks
- [ ] Multi-agent racing
- [ ] Advanced physics (tire grip, drift mechanics)

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

Made with ❤️ using Python, Pygame, and Stable-Baselines3
