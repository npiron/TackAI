# Documentation - Trackmania RL Clone

This folder contains detailed documentation for the Trackmania RL Clone project.

## 📚 Available Guides

### Training & Optimization

- **[HYPERPARAMETERS_GUIDE.md](HYPERPARAMETERS_GUIDE.md)**  
  Comprehensive guide to understanding and tuning hyperparameters for PPO training. Covers learning rate, batch size, gamma, entropy coefficient, and more.

- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)**  
  Guide to using Optuna for automated hyperparameter optimization. Learn how to run optimization trials and interpret results.

- **[ANTI_REGRESSION_GUIDE.md](ANTI_REGRESSION_GUIDE.md)**  
  Best practices to prevent training regression and maintain stable learning progress.

### Gameplay & Features

- **[GAMEPLAY_MECHANICS.md](GAMEPLAY_MECHANICS.md)**  
  Detailed explanation of game mechanics, physics, controls, and track design.

- **[REPLAY_GUIDE.md](REPLAY_GUIDE.md)**  
  How to use the replay system to save and watch personal best runs.

### Configuration

- **[HYPERPARAMS_EDITOR.md](HYPERPARAMS_EDITOR.md)**  
  Guide to editing hyperparameters and understanding the configuration files.

## 🔗 Quick Links

### Getting Started
- [Main README](../README.md) - Project overview and quick start
- [Requirements](../requirements.txt) - Python dependencies

### Scripts
- [manage.py](../manage.py) - Unified management script
- [Training Script](../rl_train.py) - RL training
- [Play Script](../rl_play.py) - Watch AI play
- [Optimization Script](../rl_optimize.py) - Hyperparameter optimization

### Configuration Files
- [rewards_config.json](../rewards_config.json) - Reward shaping configuration
- [pytest.ini](../pytest.ini) - Test configuration

## 📖 Reading Order for Beginners

If you're new to the project, we recommend reading the documentation in this order:

1. **[Main README](../README.md)** - Start here for project overview
2. **[GAMEPLAY_MECHANICS.md](GAMEPLAY_MECHANICS.md)** - Understand the game
3. **[HYPERPARAMETERS_GUIDE.md](HYPERPARAMETERS_GUIDE.md)** - Learn about training parameters
4. **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Optimize your training
5. **[ANTI_REGRESSION_GUIDE.md](ANTI_REGRESSION_GUIDE.md)** - Maintain stable training
6. **[REPLAY_GUIDE.md](REPLAY_GUIDE.md)** - Use the replay system

## 🆘 Need Help?

- Check the relevant guide above
- Run `python3 manage.py info` to see project status
- Run `python3 manage.py --help` for available commands
- Review the code comments in the source files

## 🤝 Contributing to Documentation

If you find errors or want to improve the documentation:
1. Edit the relevant `.md` file
2. Ensure formatting is consistent
3. Update this index if adding new documents
4. Submit your changes

---

[← Back to Main README](../README.md)
