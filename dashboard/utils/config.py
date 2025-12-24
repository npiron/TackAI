"""
Dashboard configuration constants.
"""

# App info
APP_TITLE = "🏎️ AI Racing Lab"
APP_DESCRIPTION = "Train neural networks to race like champions"

# Default values
DEFAULT_TRAINING_STEPS = 500_000
DEFAULT_LEARNING_RATE = 3e-4
MIN_STEPS = 100_000
MAX_STEPS = 5_000_000

# Refresh rates (seconds)
REFRESH_RATE_FAST = 1
REFRESH_RATE_NORMAL = 2
REFRESH_RATE_SLOW = 5

# UI Colors
COLORS = {
    "primary": "#10b981",    # Emerald
    "secondary": "#6366f1",  # Indigo
    "success": "#22c55e",    # Green
    "warning": "#f59e0b",    # Amber
    "error": "#ef4444",      # Red
    "background": "#1e1e2e", # Dark
}

# Training presets
TRAINING_PRESETS = {
    "Quick Test": {"steps": 100_000, "lr": 3e-4, "description": "Fast iteration (~2 min)"},
    "Standard": {"steps": 500_000, "lr": 3e-4, "description": "Balanced (~10 min)"},
    "Quality": {"steps": 1_000_000, "lr": 2e-4, "description": "Good results (~20 min)"},
    "Production": {"steps": 2_000_000, "lr": 1e-4, "description": "Best quality (~40 min)"},
}
