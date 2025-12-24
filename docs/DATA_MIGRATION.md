# Data Organization Migration

## ✅ Completed

The project files have been reorganized from a messy `logs/` directory into a clean, structured `data/` directory.

### New Structure

```
data/
├── models/
│   ├── production/      # Production-ready models (use these!)
│   └── archive/         # Old/archived models
├── checkpoints/         # Training checkpoints (auto-saved every 50k steps)
├── monitoring/          # Training metrics CSV files
├── logs/                # Application logs
│   ├── training/        # Training session logs
│   ├── game/            # Manual gameplay logs
│   └── ai/              # AI play session logs
└── optimization/        # Hyperparameter optimization results
```

### Migration Summary

**Files Migrated:**
- ✅ 2 production models → `data/models/production/`
- ✅ 98 checkpoints → `data/checkpoints/`
- ✅ 57 monitoring CSVs → `data/monitoring/`
- ✅ 16 log files → `data/logs/`
- ✅ 1 optimization file → `data/optimization/`

**Total Size:** ~223 MB

### Updated Scripts

All scripts now use the new structure:

- **`rl_train.py`**
  - Saves checkpoints to `data/checkpoints/`
  - Saves monitoring to `data/monitoring/`
  - Reads best params from `data/optimization/`

- **`rl_play.py`**
  - Looks for models in `data/models/production/` first
  - Falls back to `data/checkpoints/` if needed

- **`rl_optimize.py`**
  - Saves results to `data/optimization/`

- **`manage.py`**
  - Updated `info` command to show data/ structure

- **`.gitignore`**
  - Excludes `data/` directory (except `.gitkeep` and `README.md`)

### Usage

#### Training
```bash
python3 manage.py train
# Checkpoints saved to: data/checkpoints/
# Monitoring saved to: data/monitoring/
```

#### Watching AI
```bash
python3 manage.py watch
# Automatically finds model in: data/models/production/
```

#### Optimization
```bash
python3 manage.py optimize --trials 50
# Results saved to: data/optimization/best_hyperparams.txt
```

#### Using Optimized Hyperparameters
```bash
python3 manage.py train --use-best-params
# Reads from: data/optimization/best_hyperparams.txt
```

### Maintenance

#### Clean Old Checkpoints
```bash
# Remove checkpoints older than 30 days
find data/checkpoints -name "*.zip" -mtime +30 -delete

# Keep only last 10 checkpoints
ls -t data/checkpoints/*.zip | tail -n +11 | xargs rm
```

#### Archive Current Model
```bash
# Before training a new model
cp data/models/production/ppo_timeattack.zip \
   data/models/archive/ppo_timeattack_$(date +%Y%m%d).zip
```

#### Clean Old Logs Directory
```bash
# Interactive cleanup script
bash scripts/migrate_old_logs.sh
```

### Benefits

1. **Clear Organization** - Files grouped by purpose
2. **Easy Cleanup** - Delete old checkpoints without affecting production
3. **Better Git** - All generated files in one ignored directory
4. **Documentation** - Each directory has clear purpose
5. **Scalability** - Easy to add new data types

### Documentation

See `data/README.md` for detailed information about each directory.

---

**Migration Date:** 2025-12-22  
**Status:** ✅ Complete
