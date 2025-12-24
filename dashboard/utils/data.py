"""
Dashboard utilities for data loading and parsing.
"""
import os
import re
import glob
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict


def get_models() -> List[Dict]:
    """Get all trained models with metadata."""
    models = []
    
    for cp in glob.glob("logs/*.zip"):
        try:
            stat = os.stat(cp)
            name = os.path.basename(cp).replace(".zip", "")
            
            # Parse steps from filename
            steps_match = re.search(r'(\d+)_steps', name)
            steps = int(steps_match.group(1)) if steps_match else 0
            
            models.append({
                "name": name,
                "steps": steps,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024*1024),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "path": cp
            })
        except Exception:
            continue
    
    return sorted(models, key=lambda x: x['modified'], reverse=True)


def get_models_dataframe() -> pd.DataFrame:
    """Get models as a pandas DataFrame for display."""
    models = get_models()
    
    if not models:
        return pd.DataFrame(columns=["Model", "Steps", "Size", "Created"])
    
    return pd.DataFrame([{
        "Model": m['name'],
        "Steps": f"{m['steps']:,}",
        "Size": f"{m['size_mb']:.1f} MB",
        "Created": m['modified'].strftime("%Y-%m-%d %H:%M")
    } for m in models])


def read_log(filename: str, lines: int = 100) -> str:
    """Read last N lines from a log file."""
    if not os.path.exists(filename):
        return "No logs yet..."
    
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return "[Error reading log]"


def parse_training_metrics(log_content: str) -> Dict:
    """Parse training metrics from log content."""
    metrics = {
        "fps": 0,
        "timesteps": 0,
        "reward": 0.0,
        "episodes": 0,
        "value_loss": 0.0,
        "policy_loss": 0.0,
    }
    
    try:
        patterns = {
            "fps": r"\|\s*fps\s*\|\s*(\d+)",
            "timesteps": r"\|\s*total_timesteps\s*\|\s*(\d+)",
            "reward": r"\|\s*ep_rew_mean\s*\|\s*([\d\.-]+)",
            "episodes": r"\|\s*ep_len_mean\s*\|\s*([\d\.]+)",
            "value_loss": r"\|\s*value_loss\s*\|\s*([\d\.e\-]+)",
            "policy_loss": r"\|\s*policy_loss\s*\|\s*([\d\.e\-]+)",
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, log_content)
            if matches:
                val = matches[-1]
                if key in ["fps", "timesteps"]:
                    metrics[key] = int(val)
                else:
                    metrics[key] = float(val)
    except Exception:
        pass
    
    return metrics


def load_monitor_data(run_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load monitor CSV data, optionally for a specific run."""
    if run_id:
        csv_files = glob.glob(f"logs/monitor_*_{run_id}.csv*")
    else:
        csv_files = glob.glob("logs/monitor_*.csv*")
    
    if not csv_files:
        return None
    
    best_df = None
    max_len = 0
    
    for f in csv_files:
        try:
            df = pd.read_csv(f, skiprows=1, on_bad_lines='skip')
            if 'r' in df.columns and len(df) > max_len:
                df['r'] = pd.to_numeric(df['r'], errors='coerce')
                df = df.dropna(subset=['r'])
                if len(df) > 0:
                    best_df = df
                    max_len = len(df)
        except Exception:
            continue
    
    return best_df


def get_training_stats() -> Dict:
    """Get aggregate training statistics."""
    df = load_monitor_data()
    
    if df is None or len(df) == 0:
        return {
            "total_episodes": 0,
            "best_reward": 0.0,
            "avg_reward": 0.0,
            "recent_reward": 0.0
        }
    
    return {
        "total_episodes": len(df),
        "best_reward": df['r'].max(),
        "avg_reward": df['r'].mean(),
        "recent_reward": df['r'].tail(10).mean() if len(df) >= 10 else df['r'].mean()
    }


def delete_model(model_name: str) -> bool:
    """Delete a model by name."""
    models = get_models()
    model = next((m for m in models if m['name'] == model_name), None)
    
    if model:
        try:
            os.remove(model['path'])
            return True
        except Exception:
            return False
    return False


def clear_logs():
    """Clear all log files."""
    for f in glob.glob("*.log"):
        try:
            os.remove(f)
        except Exception:
            pass
