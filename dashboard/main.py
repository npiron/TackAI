"""
AI Racing Lab Dashboard - Dark Mode
Sidebar with clickable process buttons
"""
import gradio as gr
import subprocess
import time
import os
import signal
import sys
import psutil
import glob
import re
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === DARK MODE CSS ===
CUSTOM_CSS = """
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #0f141b;
  --border-default: #263040;
  --border-strong: #30363d;
  --text-primary: #c9d1d9;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --success: #2ea043;
  --danger: #f85149;
  --shadow-soft: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.gradio-container, .main, body, html {
  background: var(--bg-primary) !important;
  outline: none !important;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.panel, .block, .form, .container {
  background: var(--bg-secondary) !important;
  border-color: var(--border-strong) !important;
  outline: none !important;
  border-radius: 14px !important;
}

*, label, span, p, h1, h2, h3, h4, .label, .label-wrap {
  color: var(--text-primary) !important;
}

h1, h2, h3 {
  color: var(--accent) !important;
  letter-spacing: 0.2px;
}

.gradio-container .tabitem {
  gap: 16px;
}

input, textarea, select, .wrap {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: 10px !important;
}

table, th, td, .dataframe {
  background: var(--bg-secondary) !important;
  color: var(--text-primary) !important;
  border-color: var(--border-default) !important;
}

th {
  background: #21262d !important;
}

button {
  background: #21262d !important;
  border: 1px solid var(--border-default) !important;
  color: var(--text-primary) !important;
  border-radius: 12px !important;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
  transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}

button:hover {
  background: #30363d !important;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.25);
  transform: translateY(-1px);
}

button.primary {
  background: linear-gradient(135deg, #238636, var(--success)) !important;
  border-color: #238636 !important;
  color: #f0f6fc !important;
}

button.stop {
  background: linear-gradient(135deg, #da3633, var(--danger)) !important;
  border-color: #da3633 !important;
  color: #f0f6fc !important;
}

.sidebar {
  background: linear-gradient(180deg, #0d1117, #141a23) !important;
  border-right: 1px solid #21262d !important;
}

.gr-number input {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
}

.proc-btn {
  text-align: left !important;
  justify-content: flex-start !important;
}

.stat-card .block,
.panel-card .block {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border-default) !important;
  box-shadow: var(--shadow-soft);
  padding: 14px 16px !important;
}

.stat-card .label,
.panel-card .label {
  color: var(--text-muted) !important;
  font-weight: 600 !important;
}

.mono textarea,
.mono input {
  font-family: "JetBrains Mono", "SFMono-Regular", Menlo, monospace !important;
}

.section-divider {
  margin: 12px 0 6px;
  border-top: 1px solid var(--border-default);
}

/* Remove focus outlines */
*:focus {
  outline: none !important;
  box-shadow: none !important;
}
.gradio-container:focus, .main:focus { outline: none !important; }
"""

# === UTILITIES ===

def get_python_cmd():
    venv = os.path.abspath(".venv/bin/python")
    return venv if os.path.exists(venv) else sys.executable

def get_running_processes():
    procs = []
    keywords = {
        "rl_train.py": ("🧠 Training", "training"), 
        "rl_play.py": ("🤖 AI Play", "ai"), 
        "trackmania_clone.py": ("🎮 Manual", "game"),
        "rl_optimize.py": ("🧪 Optimization", "optimization")
    }
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and "python" in proc.info['name'].lower():
                for arg in cmdline:
                    for kw, (label, ptype) in keywords.items():
                        if kw in arg:
                            procs.append({
                                "pid": proc.info['pid'],
                                "label": label,
                                "type": ptype,
                                "start_time": proc.info['create_time']
                            })
        except: pass
    return procs

def format_time_duration(seconds):
    if seconds < 60: return f"{int(seconds)}s"
    elif seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def format_time(start_time):
    return format_time_duration(time.time() - start_time)

def get_training_info(pid=None):
    total_steps = 2_000_000 # Default fallback
    run_id = None
    
    # Try to get real target from API or process
    if pid:
        try:
            p = psutil.Process(int(pid))
            cmd = p.cmdline()
            if "--steps" in cmd:
                idx = cmd.index("--steps")
                if idx + 1 < len(cmd):
                    total_steps = int(float(cmd[idx+1]))
            
            if "--run-id" in cmd:
                idx = cmd.index("--run-id")
                if idx + 1 < len(cmd):
                    run_id = cmd[idx+1]
        except: pass

    current = 0
    fps = 0
    eta = "N/A"
    
    try:
        # Read log for current steps and FPS
        # If we have a run_id, construct the path directly
        if run_id:
            log_file = f"data/logs/training/{run_id}.log"
        else:
            log_file = get_latest_log("training")

        if log_file and os.path.exists(log_file):
             with open(log_file, 'r', errors='replace') as f:
                 # Read last 20KB
                 f.seek(0, 2)
                 size = f.tell()
                 f.seek(max(0, size - 20000), 0)
                 content = f.read()
                 
                 steps = re.findall(r"\|\s*total_timesteps\s*\|\s*(\d+)", content)
                 if steps: current = int(steps[-1])
                     
                 fps_m = re.findall(r"\|\s*fps\s*\|\s*(\d+)", content)
                 if fps_m: fps = int(fps_m[-1])
                 
                 if pid is None:
                     target = re.search(r"Training for (\d+) steps", content)
                     if target: total_steps = int(target.group(1))

    except: pass
    
    # Calc progress
    if total_steps <= 0: total_steps = 1
    progress = min(100, int(100 * current / total_steps))
    
    if fps > 0 and current < total_steps:
        eta = format_time_duration((total_steps - current) / fps)
        
    return {
        "progress": progress,
        "current": current,
        "total": total_steps,
        "fps": fps,
        "eta": eta,
        "run_id": run_id
    }

def get_training_progress(pid=None):
    return get_training_info(pid)["progress"]

def get_models():
    models = []
    
    # Search paths: production models, checkpoints, and legacy logs
    patterns = [
        "data/models/production/*.zip",
        "data/checkpoints/*.zip",
        "logs/*.zip"
    ]
    
    model_files = []
    for p in patterns:
        model_files.extend(glob.glob(p))
        
    for cp in model_files:
        try:
            stat = os.stat(cp)
            name = os.path.basename(cp).replace(".zip", "")
            
            # Try to extract steps and run_id
            steps = 0
            run_id = name
            
            steps_match = re.search(r'^(.*)_(\d+)_steps$', name)
            if steps_match:
                run_id = steps_match.group(1)
                steps = int(steps_match.group(2))
            elif name.endswith("_final"):
                run_id = name[:-6] # remove _final
                steps = 999_999_999 
            elif "ppo" in name: # Production models often have simple names
                 steps = 999_999_999
            
            # Find stats from monitor files matches run_id
            # Log file pattern: monitor_headless_{run_id}_{pid}.csv or monitor_visual_{run_id}.csv
            # We look for *{run_id}*.csv but specifically in the monitor format
            
            success_rate = 0.0
            best_lap = 0.0
            
            # Look in monitoring dir and legacy logs
            monitor_patterns = [
                f"data/monitoring/monitor_*_{run_id}_*.csv",
                f"data/monitoring/monitor_*_{run_id}.csv",
                f"logs/monitor_*_{run_id}_*.csv",
                f"logs/monitor_*_{run_id}.csv"
            ]
            
            monitor_files = []
            for mp in monitor_patterns:
                monitor_files.extend(glob.glob(mp))
            
            if monitor_files:
                try:
                    dfs = []
                    for mf in monitor_files:
                        try:
                            # Skip first line comment/metadata usually handled by stable_baselines3 monitor, but pandas needs help
                            d = pd.read_csv(mf, comment='#')
                            if not d.empty:
                                dfs.append(d)
                        except: pass
                    
                    if dfs:
                        full_df = pd.concat(dfs)
                        if 'is_success' in full_df.columns:
                            # Last 100 episodes
                            success_rate = pd.to_numeric(full_df['is_success'], errors='coerce').tail(100).mean() * 100
                        if 'lap_time' in full_df.columns and 'is_success' in full_df.columns:
                            # Filter for successes
                            successes = full_df[full_df['is_success'] > 0]
                            if len(successes) > 0:
                                best_lap = pd.to_numeric(successes['lap_time'], errors='coerce').min()
                except: pass

            models.append({
                "name": name, 
                "steps": steps, 
                "size": stat.st_size/(1024*1024), 
                "date": datetime.fromtimestamp(stat.st_mtime), 
                "path": cp,
                "success": success_rate,
                "best_lap": best_lap
            })
        except: continue
    return sorted(models, key=lambda x: x['date'], reverse=True)

def get_models_df():
    models = get_models()
    if not models:
        return pd.DataFrame(columns=["Model", "Steps", "Success", "Best Lap", "Size", "Modified"])
    
    rows = []
    for m in models:
        step_str = "Final" if m['steps'] == 999_999_999 else f"{m['steps']:,}"
        best_str = f"{m['best_lap']:.2f}s" if m['best_lap'] > 0 else "--"
        rows.append({
            "Model": m['name'], 
            "Steps": step_str,
            "Success": f"{m['success']:.0f}%",
            "Best Lap": best_str,
            "Size": f"{m['size']:.1f} MB", 
            "Modified": m['date'].strftime("%Y-%m-%d %H:%M")
        })
    return pd.DataFrame(rows)

def read_log(filename, lines=40):
    if os.path.exists(filename):
        try:
            with open(filename, "r", errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except: return "[Error]"
    return "No log..."

def get_latest_log(prefix):
    # Mapping prefixes to directories
    # For training logs, we accept ANY .log file in the training dir, 
    # because user might have named their run anything.
    
    patterns = []
    
    if prefix == "training":
        patterns = ["data/logs/training/*.log", "logs/training_*.log"]
    elif prefix == "optimization":
        patterns = ["data/logs/optimization/*.log", "logs/optimization_*.log", "data/logs/training/optimization_*.log"] # checks old locations too
    elif prefix == "game":
        patterns = ["data/logs/game/*.log", "logs/game_*.log"]
    elif prefix == "ai":
        patterns = ["data/logs/ai/*.log", "logs/ai_*.log"]
    else:
        # Fallback
        patterns = [f"logs/{prefix}_*.log"]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    # Filter out optimization logs from training logs if they ended up there by mistake previously
    if prefix == "training":
        files = [f for f in files if "optimization_" not in os.path.basename(f)]

    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def get_system_stats():
    return psutil.cpu_percent(interval=0.1), psutil.virtual_memory().percent

def load_monitor_data(run_id=None):
    # Look in new monitoring dir AND old logs dir for backward compatibility
    files = sorted(glob.glob("data/monitoring/monitor_*.csv*") + glob.glob("logs/monitor_*.csv*"), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    
    target_files = []
    
    if run_id:
        # Exact match filtering
        target_files = [f for f in files if run_id in os.path.basename(f)]
    else:
        # Fallback to latest file strategy
        latest_file = files[0]
        filename = os.path.basename(latest_file)
        
        # Determine run_id from latest file
        clean = filename.replace("monitor_headless_", "").replace("monitor_visual_", "")
        while clean.endswith(".csv") or clean.endswith(".monitor"):
            if clean.endswith(".csv"): clean = clean[:-4]
            if clean.endswith(".monitor"): clean = clean[:-8]
            
        detected_run_id = clean
        match_pid = re.search(r"^(.*)_(\d+)$", clean)
        if match_pid:
            detected_run_id = match_pid.group(1)
            
        target_files = [f for f in files if detected_run_id in os.path.basename(f)]
    
    if not target_files:
        return None
        
    dfs = []
    for f in target_files:
        try:
            df = pd.read_csv(f, skiprows=1, on_bad_lines='skip')
            if 'r' in df.columns:
                df['r'] = pd.to_numeric(df['r'], errors='coerce')
                dfs.append(df.dropna(subset=['r']))
        except: continue
    
    if not dfs:
        return None
    
    # Filter out empty dataframes to avoid FutureWarning
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        return None
        
    return pd.concat(dfs).sort_values('t')

def parse_metrics(log):
    m = {"fps": 0, "steps": 0, "reward": 0.0}
    try:
        fps = re.findall(r"\|\s*fps\s*\|\s*(\d+)", log)
        steps = re.findall(r"\|\s*total_timesteps\s*\|\s*(\d+)", log)
        rew = re.findall(r"\|\s*ep_rew_mean\s*\|\s*([\d\.-]+)", log)
        if fps: m["fps"] = int(fps[-1])
        if steps: m["steps"] = int(steps[-1])
        if rew: m["reward"] = float(rew[-1])
    except: pass
    return m

def get_chart_data(run_id=None):
    df = load_monitor_data(run_id)
    if df is None or len(df) < 5:
        return pd.DataFrame({"Episode": [0], "Value": [0], "Metric": ["Reward"]})
    
    # Create Episode index
    df = df.reset_index(drop=True)
    df['Episode'] = df.index
    
    # Select cols
    cols = {
        'r': 'Reward',
        'l': 'Length',
        'is_success': 'Success Rate (%)',
        'checkpoints_reached': 'Checkpoints'
    }
    
    # Process valid columns
    valid_cols = [c for c in cols.keys() if c in df.columns]
    
    if not valid_cols:
         return pd.DataFrame({"Episode": [0], "Value": [0], "Metric": ["Reward"]})
         
    # Compute Rolling Means (Window 50)
    # We want to smooth specific columns
    smooth_df = df[['Episode']].copy()
    
    for c in valid_cols:
        series = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Special logic for Success (0/1 -> %)
        if c == 'is_success':
            smooth_df[cols[c]] = series.rolling(window=50, min_periods=1).mean() * 100
        else:
            smooth_df[cols[c]] = series.rolling(window=50, min_periods=1).mean()
            
    # Melt to Long Format for Multi-Line Plotting
    plot_df = smooth_df.melt(id_vars=['Episode'], var_name='Metric', value_name='Value')
    
    # Drop NaNs created by rolling window to avoid plot issues
    plot_df = plot_df.dropna()
    
    # Limit size for performance (Last 300 points per metric for smooth UI)
    # We can just take tail(300 * num_metrics)
    n_metrics = len(valid_cols)
    return plot_df.tail(300 * n_metrics)

def get_advanced_stats(run_id=None):
    df = load_monitor_data(run_id)
    if df is None or len(df) < 5:
        # Must return 5 values to match unpacking
        return 0, "0%", "N/A", "--", 0
    
    # Defaults
    avg_cp = 0.0
    success_rate = 0.0
    avg_lap = 0.0
    best_lap = 0.0
    max_cp = 0
    
    # Ensure numeric columns to prevent TypeErrors
    if 'checkpoints_reached' in df.columns:
        df['checkpoints_reached'] = pd.to_numeric(df['checkpoints_reached'], errors='coerce').fillna(0)
        avg_cp = df['checkpoints_reached'].tail(50).mean()
        max_cp = int(df['checkpoints_reached'].max())

    if 'is_success' in df.columns:
        df['is_success'] = pd.to_numeric(df['is_success'], errors='coerce').fillna(0)
        success_rate = df['is_success'].tail(50).mean() * 100

    if 'lap_time' in df.columns and 'is_success' in df.columns:
        df['lap_time'] = pd.to_numeric(df['lap_time'], errors='coerce').fillna(0.0)
        successes = df[df['is_success'] > 0]
        if len(successes) > 0:
            avg_lap = successes['lap_time'].tail(20).mean()
            best_lap = successes['lap_time'].min()

    lap_str = f"{avg_lap:.1f}s" if avg_lap > 0 else "N/A"
    best_str = f"{best_lap:.2f}s" if best_lap > 0 else "--"
    return round(avg_cp, 1), f"{success_rate:.0f}%", lap_str, best_str, max_cp

def load_rewards(pid=None):
    default = {
        "progress_weight": 4.0,
        "track_penalty": 2.0,
        "speed_weight": 1.5,
        "wall_penalty": 3.0,
        "checkpoint_bonus": 100.0,
        "time_bonus": 200.0,
        "alignment_weight": 0.5,
        "checkpoint_scaling": 0.0,
        "ghost_following_weight": 0.0
    }
    
    # Load base config
    if os.path.exists("rewards_config.json"):
        try:
            import json
            with open("rewards_config.json", "r") as f:
                data = json.load(f)
                default.update(data)
        except: pass

    # Load specific config if PID provided
    if pid:
        specific = f"rewards_config_{pid}.json"
        if os.path.exists(specific):
            try:
                import json
                with open(specific, "r") as f:
                    data = json.load(f)
                    default.update(data)
            except: pass
            
    return default

def save_rewards(pw, tp, sw, wp, cb, tb, aw, cs, gfw, pid=None):
    import json
    data = {
        "progress_weight": pw,
        "track_penalty": tp,
        "speed_weight": sw,
        "wall_penalty": wp,
        "checkpoint_bonus": cb,
        "time_bonus": tb,
        "alignment_weight": aw,
        "checkpoint_scaling": cs,
        "ghost_following_weight": gfw
    }
    
    target_file = "rewards_config.json"
    msg = "✅ Saved Globally! All AIs will update."
    
    if pid:
        target_file = f"rewards_config_{pid}.json"
        msg = f"✅ Saved for PID {pid}! AI will update."
        
    with open(target_file, "w") as f:
        json.dump(data, f, indent=4)
        
    return msg

def save_hyperparams(nsteps, batch, gamma, gae, ent, lr, clip, grad, vf):
    import json
    data = {
        "n_steps": int(nsteps),
        "batch_size": int(batch),
        "gamma": float(gamma),
        "gae_lambda": float(gae),
        "ent_coef": float(ent),
        "learning_rate": float(lr),
        "clip_range": float(clip),
        "max_grad_norm": float(grad),
        "vf_coef": float(vf)
    }
    os.makedirs("data/optimization", exist_ok=True)
    with open("data/optimization/best_hyperparams.txt", "w") as f:
        json.dump(data, f, indent=4)
    return "✅ Hyperparameters saved!"

def preset_stable_params():
    return 2048, 256, 0.995, 0.95, 0.005, 0.0003, 0.2, 0.5, 0.5

def preset_fast_params():
    return 1024, 512, 0.995, 0.95, 0.01, 0.0005, 0.2, 0.5, 0.5

def preset_finetune_params():
    return 2048, 128, 0.995, 0.95, 0.001, 0.0001, 0.15, 0.3, 0.5

# === ACTIONS ===

def start_training(steps, custom_name, base_model, use_opt, visual_mode, procedural):
    # Determine Run ID (name)
    if custom_name and custom_name.strip():
        # Sanitize user input
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', custom_name.strip())
        run_id = f"{safe_name}_{int(time.time())}"
    else:
        run_id = f"training_{int(time.time())}"
        
    log = f"data/logs/training/{run_id}.log"
    os.makedirs("data/logs/training", exist_ok=True)
    
    cmd = [get_python_cmd(), "-u", "rl_train.py", "--steps", str(int(steps)), "--run-id", run_id]
    
    if use_opt:
        cmd.append("--use-best-params")
        
    if visual_mode:
        cmd.append("--visual")

    if not procedural:
        cmd.append("--static")
    
    # Continue training from model?
    if base_model and base_model != "None (New Model)":
        m = next((x for x in get_models() if x['name'] == base_model), None)
        if m:
            cmd.extend(["--load", m['path']])
    
    with open(log, "w") as f:
        f.write(f"Training started: {run_id}\nSteps: {int(steps)}\nBase Model: {base_model}\n")
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    return f"✅ Started: {run_id}"

def start_ai_with_model(model_name, procedural=True):
    if not model_name: return "❌ Select a model"
    m = next((x for x in get_models() if x['name'] == model_name), None)
    if not m: return "❌ Model not found"
    if not m: return "❌ Model not found"
    # Don't copy anymore! Pass path directly.
    os.makedirs("data/logs/ai", exist_ok=True)
    log = f"data/logs/ai/ai_{int(time.time())}.log"
    
    cmd = [get_python_cmd(), "rl_play.py", "--model", m['path']]
    
    # Try to find matching env file
    # Pattern 1: {name}_vecnormalize.pkl (if name is like "myrun_final")
    # Pattern 2: logs/{name}_vecnormalize.pkl
    # Pattern 3: defaults
    
    env_path = None
    
    # Try replacing _final with _vecnormalize
    base_name = m['name'].replace("_final", "")
    candidate = f"logs/{base_name}_vecnormalize.pkl"
    if os.path.exists(candidate):
        env_path = candidate
    
    # Fallbacks
    if not env_path and os.path.exists("logs/vecnormalize.pkl"):
        env_path = "logs/vecnormalize.pkl"
    elif not env_path and os.path.exists("vecnormalize.pkl"):
        env_path = "vecnormalize.pkl"
        
    if env_path:
        cmd.extend(["--env", env_path])
        
    if not procedural:
        cmd.append("--no-procedural")
        
    with open(log, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return f"✅ AI started"

def start_game():
    os.makedirs("data/logs/game", exist_ok=True)
    log = f"data/logs/game/game_{int(time.time())}.log"
    with open(log, "w") as f:
        subprocess.Popen([get_python_cmd(), "trackmania_clone.py"], stdout=f, stderr=subprocess.STDOUT)
    return "✅ Game launched"

def start_optimization(n_trials):
    os.makedirs("data/logs/optimization", exist_ok=True)
    log = f"data/logs/optimization/optimization_{int(time.time())}.log"
    with open(log, "w") as f:
        # Pass number of trials
        subprocess.Popen([get_python_cmd(), "-u", "rl_optimize.py", "--trials", str(int(n_trials))], stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5) # Ensure file exists
    return f"✅ Optimization started in background ({n_trials} trials)"

def kill_process(pid):
    try:
        os.kill(int(pid), signal.SIGINT)
        return f"✅ Killed {pid}"
    except Exception as e:
        return f"❌ {e}"

def stop_all():
    c = 0
    for p in get_running_processes():
        try: os.kill(p['pid'], signal.SIGINT); c += 1
        except: pass
    return f"⏹️ Stopped {c}"

def delete_model(name):
    if not name: return "❌ Select", get_models_df()
    m = next((x for x in get_models() if x['name'] == name), None)
    if m:
        os.remove(m['path'])
        return f"🗑️ Deleted", get_models_df()
    return "❌ Not found", get_models_df()

# === UI ===

with gr.Blocks(title="🏎️ AI Racing Lab", theme=gr.themes.Base()) as app:
    
    # State
    current_view = gr.State("dashboard")
    selected_pid = gr.State(None)
    process_refresh = gr.State(0)  # Trigger for process list refresh
    
    with gr.Row(equal_height=True):
        
        # === SIDEBAR ===
        with gr.Column(scale=1, min_width=280, elem_classes=["sidebar"]):
            gr.Markdown("# 🏎️ Racing Lab")
            dashboard_btn = gr.Button("🏠 Dashboard", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### ⚡ Actions")
            with gr.Row():
                new_train_btn = gr.Button("🧠 Train", size="sm")
                tuning_btn = gr.Button("🔧 Tune", size="sm")
            with gr.Row():
                hyperparams_btn = gr.Button("🎛️ Hyperparams", size="sm")
                opt_view_btn = gr.Button("✨ New Opt.", size="sm")
            play_manual_btn = gr.Button("🎮 Play", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("### 💻 System")
            with gr.Row():
                cpu_txt = gr.Textbox(value="0%", label="CPU", interactive=False)
                ram_txt = gr.Textbox(value="0%", label="RAM", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📡 Processes")
            
            # Dynamic process buttons using @gr.render
            @gr.render(inputs=[process_refresh])
            def render_processes(refresh_count):
                procs = get_running_processes()
                if not procs:
                    gr.Markdown("*No running processes*")
                else:
                    for p in procs:
                        prog = get_training_progress(p['pid']) if p['type'] == 'training' else None
                        prog_txt = f" ({prog}%)" if prog is not None else ""
                        with gr.Row():
                            # Process button - click to view details
                            btn = gr.Button(
                                f"{p['label']} - {format_time(p['start_time'])}{prog_txt}",
                                size="sm",
                                scale=4,
                                elem_classes=["proc-btn"]
                            )
                            # Kill button
                            kill_btn = gr.Button("🗑️", size="sm", variant="stop", scale=1)
                            
                            # Wire up handlers
                            def make_view_handler(pid, ptype):
                                def handler():
                                    if ptype == 'training':
                                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "training", pid
                                    elif ptype == 'optimization':
                                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "optimization", pid
                                    else:
                                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "process", pid
                                return handler
                            
                            def make_kill_handler(pid):
                                def handler():
                                    return kill_process(pid)
                                return handler
                            
                            btn.click(make_view_handler(p['pid'], p['type']), 
                                     outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
                            kill_btn.click(make_kill_handler(p['pid']), outputs=[action_status])
            
            gr.Markdown("---")
            stop_all_btn = gr.Button("⏹️ Stop All", variant="stop", size="sm")
            action_status = gr.Textbox(label="Status", interactive=False, lines=1)
        
        # === MAIN CONTENT ===
        with gr.Column(scale=4):
            
            # DASHBOARD VIEW
            with gr.Column(visible=True) as dashboard_view:
                gr.Markdown("## 📊 Dashboard")
                
                with gr.Row():
                    n_models = gr.Number(value=len(get_models()), label="📦 Models", interactive=False)
                    df = load_monitor_data()
                    n_eps = gr.Number(value=len(df) if df is not None else 0, label="🎯 Episodes", interactive=False)
                    best_rew = gr.Number(value=round(df['r'].max(), 1) if df is not None and len(df) > 0 else 0, label="🏆 Best", interactive=False)
                    n_procs = gr.Number(value=len(get_running_processes()), label="📡 Active", interactive=False)
                
                gr.Markdown("### 📦 Models")
                models_table = gr.Dataframe(value=get_models_df(), interactive=False, max_height=320)
                
                with gr.Row():
                    model_select = gr.Dropdown(choices=[m['name'] for m in get_models()], label="Model", scale=3)
                    chk_play_procedural = gr.Checkbox(label="Procedural", value=True)
                    play_ai_btn = gr.Button("▶️ Play AI", variant="primary")
                    del_model_btn = gr.Button("🗑️ Delete", variant="stop")
                
                refresh_btn = gr.Button("🔄 Refresh")
            
            # TRAINING VIEW
            with gr.Column(visible=False) as training_view:
                gr.Markdown("## 🧠 Training")
                
                with gr.Row():
                    t_fps = gr.Number(value=0, label="⚡ FPS", interactive=False)
                    t_steps = gr.Number(value=0, label="📈 Steps", interactive=False)
                    t_rew = gr.Number(value=0, label="🎯 Reward", interactive=False)
                    t_prog = gr.Number(value=0, label="📊 %", interactive=False)
                
                with gr.Row():
                    t_cps = gr.Number(value=0, label="🚩 Avg CPs", interactive=False)
                    t_max_cp = gr.Number(value=0, label="🏁 Max CP", interactive=False)
                    t_success = gr.Textbox(value="0%", label="🏁 Success Rate", interactive=False)
                
                with gr.Row():
                    t_lap = gr.Textbox(value="N/A", label="⏱️ Avg Lap", interactive=False)
                    t_best = gr.Textbox(value="--", label="🏆 Best Record", interactive=False)
                
                t_logs = gr.Textbox(value="", label="📜 Logs", lines=12, interactive=False)
                
                # Charts Grid
                gr.Markdown("### 📊 Live Analytics")
                
                # Main Chart: Reward
                plot_reward = gr.LinePlot(
                    x="Episode", y="Value", color="Metric", 
                    height=300, 
                    title="📈 Reward Trend (Learning Algo)",
                    tooltip=["Episode", "Value", "Metric"],
                    x_title="Episode", y_title="Score"
                )
                
                with gr.Row():
                    # Secondary: Success
                    plot_success = gr.LinePlot(
                        x="Episode", y="Value", color="Metric", 
                        height=250, 
                        title="🏁 Success Rate & Checkpoints",
                        tooltip=["Episode", "Value", "Metric"]
                    )
                    
                    # Tertiary: Length
                    plot_length = gr.LinePlot(
                        x="Episode", y="Value", color="Metric", 
                        height=250, 
                        title="⏱️ Episode Duration (Steps)",
                        tooltip=["Episode", "Value", "Metric"]
                    )
                back_train = gr.Button("← Back")
            
            # PROCESS VIEW (AI/Game)
            with gr.Column(visible=False) as process_view:
                gr.Markdown("## 📡 Process Details")
                p_info = gr.Markdown("")
                p_logs = gr.Textbox(value="", label="📜 Logs", lines=18, interactive=False)
                back_proc = gr.Button("← Back")
            
            # NEW TRAINING VIEW
            with gr.Column(visible=False) as new_training_view:
                gr.Markdown("## 🧠 New Training Session")
                
                with gr.Row():
                    custom_name_input = gr.Textbox(label="Process Name (Optional)", placeholder="e.g. experimental_reward", lines=1)
                    base_model_input = gr.Dropdown(label="Continue from Model", choices=["None (New Model)"], value="None (New Model)")
                
                steps_slider = gr.Slider(100_000, 5_000_000, 500_000, step=100_000, label="Training Steps")
                
                has_params = os.path.exists("data/optimization/best_hyperparams.txt")
                opt_info = " (Trouvé !)" if has_params else " (Pas encore trouvé)"
                
                with gr.Row():
                     use_opt_checkbox = gr.Checkbox(label=f"Utiliser les hyperparamètres optimisés {opt_info}", value=has_params, interactive=True)
                     visual_mode_checkbox = gr.Checkbox(label="Training Preview (Slower, 1 Core)", value=False, interactive=True)
                     procedural_checkbox = gr.Checkbox(label="🎲 Circuit Procédural (Aléatoire)", value=True, interactive=True)
                
                # Hyperparameters Details (collapsible)
                with gr.Accordion("📊 Détails des Hyperparamètres", open=False):
                    if has_params:
                        try:
                            import ast
                            with open("data/optimization/best_hyperparams.txt", "r") as f:
                                params = ast.literal_eval(f.read())
                            
                            gr.Markdown(f"""
**Paramètres Optimisés Actifs :**
- **Learning Rate** : `{params.get('learning_rate', 'N/A')}` (Vitesse d'apprentissage)
- **Batch Size** : `{params.get('batch_size', 'N/A')}` (Taille des lots)
- **n_steps** : `{params.get('n_steps', 'N/A')}` (Steps avant update)
- **Gamma** : `{params.get('gamma', 'N/A')}` (Discount factor)
- **Entropy Coef** : `{params.get('ent_coef', 'N/A')}` (Exploration)
- **GAE Lambda** : `{params.get('gae_lambda', 'N/A')}` (Advantage estimation)

💡 **Learning Rate Schedule** : Décroissant (5e-4 → 3e-4 → 1e-4) pour éviter la régression !
                            """)
                        except:
                            gr.Markdown("⚠️ Erreur de lecture des paramètres")
                    else:
                        gr.Markdown("""
**Paramètres par Défaut :**
- **Learning Rate** : `3e-4` (Standard)
- **Batch Size** : `256`
- **n_steps** : `2048`
- **Gamma** : `0.995`

💡 Lance **Optuna** pour trouver les meilleurs paramètres automatiquement !
                        """)
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Training", variant="primary")
                    cancel_btn = gr.Button("Cancel")
                
                train_status = gr.Textbox(label="Status", interactive=False)

            # TUNING VIEW
            with gr.Column(visible=False) as tuning_view:
                gr.Markdown("## 🔧 Reward Tuning (Live)")
                gr.Markdown("""
Ajuste les récompenses pour guider l'IA. 
**Important :** L'a cherche juste à maximiser son score. Si une pénalité est trop faible, elle l'ignorera. Si elle est trop forte, elle sera tétanisée.
**Changes apply immediately** to running training sessions.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🛑 Survie & Trajectoire (Les Freins)")
                        gr.Markdown("*Empêche l'IA de tricher ou de se crasher.*")
                        
                        r_track_penalty = gr.Slider(0.0, 10.0, step=0.1, label="Off-Track Penalty (The Floor is Lava)", 
                                                    info="Points perdus par seconde dans l'herbe. Haut = Reste sur la route (Vital). Bas = Coupe les virages.")
                        
                        r_wall_penalty = gr.Slider(0.0, 10.0, step=0.1, label="Wall Hit Penalty", 
                                                   info="Pénalité immédiate en cas de choc. Haut = Évite les murs à tout prix. Bas = Wall-riding possible.")
                                                   
                        r_align = gr.Slider(0.0, 5.0, step=0.1, label="Centerline Alignment (Slot Car)", 
                                            info="Récompense le maintien au centre de la piste et dans le bon sens. Haut = Conduite propre. Bas = Liberté totale.")

                    with gr.Column():
                        gr.Markdown("### 🚀 Moteur & Objectifs (L'Accélérateur)")
                        gr.Markdown("*Motive l'IA à avancer et à finir.*")
                        
                        r_progress = gr.Slider(0.0, 20.0, step=0.1, label="Progress Weight (La Carotte)", 
                                               info="Récompense principale pour avancer vers le but. Ne jamais mettre à 0.")
                                               
                        r_speed = gr.Slider(0.0, 5.0, step=0.1, label="Speed Weight (Vitesse)", 
                                            info="Bonus de vitesse pure. Attention : Si > Off-Track, l'IA foncera dans le mur.")
                                            
                        r_cp_bonus = gr.Slider(0, 500, step=10, label="Checkpoint Bonus (Ballise)", 
                                               info="Gros bonus unique à chaque Checkpoint validé. Indispensable pour les circuits complexes.")
                                               
                        r_scaling = gr.Slider(0.0, 2.0, step=0.05, label="Progressive Scaling (Endurance)", 
                                              info="Multiplicateur de bonus (+X%) à chaque CP validé. Pousse l'IA à finir le tour.")
                                              
                        r_time_bonus = gr.Slider(0, 1000, step=10, label="Finish Time Bonus", 
                                                 info="Bonus final basé sur le temps restant. Pour l'optimisation ultime.")
                        
                        r_ghost_follow = gr.Slider(0.0, 10.0, step=0.1, label="Ghost Following (Static Tracks Only)", 
                                                   info="Récompense pour suivre la trajectoire du PB. Uniquement sur circuits fixes (--static). 0 = Désactivé.")
                
                save_tuning_btn = gr.Button("💾 Save Configuration", variant="primary")
                tuning_status = gr.Textbox(label="Status", interactive=False) 
                
                back_tuning = gr.Button("← Back")

            # HYPERPARAMETERS EDITOR VIEW (DQN)
            with gr.Column(visible=False) as hyperparams_view:
                gr.Markdown("## 🎛️ DQN Hyperparameters Editor")
                gr.Markdown("""
Ajuste les hyperparamètres **DQN** pour contrôler **comment** l'IA apprend.
Ces paramètres sont utilisés quand tu coches "Utiliser les hyperparamètres optimisés".

**DQN** = Deep Q-Network (Off-Policy, Replay Buffer)
                """)
                
                # Load current params
                def load_hyperparams():
                    default = {
                        "learning_rate": 0.0003,
                        "batch_size": 256,
                        "buffer_size": 200000,
                        "gamma": 0.99,
                        "learning_starts": 5000,
                        "target_update_interval": 1000,
                        "exploration_fraction": 0.1,
                        "exploration_final_eps": 0.05,
                        "max_grad_norm": 10.0
                    }
                    try:
                        import ast
                        with open("data/optimization/best_hyperparams.txt", "r") as f:
                            default.update(ast.literal_eval(f.read()))
                    except: pass
                    return default
                
                # Presets
                with gr.Row():
                    gr.Markdown("### 📋 Presets DQN")
                with gr.Row():
                    preset_stable = gr.Button("🛡️ Stable (Recommandé)", size="sm")
                    preset_fast = gr.Button("⚡ Fast Learning", size="sm")
                    preset_conservative = gr.Button("🎯 Conservative", size="sm")
                
                gr.Markdown("---")
                
                # Sliders
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🧠 Learning")
                        h_lr = gr.Slider(0.00001, 0.001, step=0.00001, value=0.0003, label="Learning Rate", 
                                        info="Vitesse d'apprentissage (avec schedule décroissant)")
                        h_batch = gr.Slider(32, 512, step=32, value=256, label="Batch Size", 
                                           info="Taille des lots pour chaque update")
                        h_buffer = gr.Slider(50000, 500000, step=50000, value=200000, label="Buffer Size", 
                                            info="Taille du replay buffer (mémoire)")
                    
                    with gr.Column():
                        gr.Markdown("### 🎯 Q-Learning")
                        h_gamma = gr.Slider(0.9, 0.999, step=0.001, value=0.99, label="Gamma (Discount)", 
                                           info="Importance du long-terme (0.99 = très patient)")
                        h_learn_starts = gr.Slider(1000, 20000, step=1000, value=5000, label="Learning Starts", 
                                                   info="Steps avant de commencer l'apprentissage")
                        h_target_update = gr.Slider(500, 5000, step=500, value=1000, label="Target Update Interval", 
                                                    info="Fréquence de mise à jour du réseau cible")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔍 Exploration")
                        h_expl_frac = gr.Slider(0.05, 0.3, step=0.05, value=0.1, label="Exploration Fraction", 
                                               info="% du training en exploration (10% = 100k/1M steps)")
                        h_expl_final = gr.Slider(0.01, 0.2, step=0.01, value=0.05, label="Final Epsilon", 
                                                info="% d'actions aléatoires en fin de training")
                    
                    with gr.Column():
                        gr.Markdown("### 🔒 Stability")
                        h_grad = gr.Slider(1.0, 20.0, step=1.0, value=10.0, label="Max Grad Norm", 
                                          info="Limite les changements brusques (gradient clipping)")
                
                save_hyperparams_btn = gr.Button("💾 Save Hyperparameters", variant="primary")
                hyperparams_status = gr.Textbox(label="Status", interactive=False)
                
                back_hyperparams = gr.Button("← Back")
                
                # Preset handlers
                def apply_stable():
                    return 0.0003, 256, 200000, 0.99, 5000, 1000, 0.1, 0.05, 10.0, "✅ Preset 'Stable' appliqué"
                
                def apply_fast():
                    return 0.0005, 128, 100000, 0.98, 2000, 500, 0.15, 0.1, 5.0, "✅ Preset 'Fast Learning' appliqué"
                
                def apply_conservative():
                    return 0.0001, 512, 300000, 0.995, 10000, 2000, 0.05, 0.02, 15.0, "✅ Preset 'Conservative' appliqué"
                
                preset_stable.click(apply_stable, outputs=[h_lr, h_batch, h_buffer, h_gamma, h_learn_starts, h_target_update, h_expl_frac, h_expl_final, h_grad, hyperparams_status])
                preset_fast.click(apply_fast, outputs=[h_lr, h_batch, h_buffer, h_gamma, h_learn_starts, h_target_update, h_expl_frac, h_expl_final, h_grad, hyperparams_status])
                preset_conservative.click(apply_conservative, outputs=[h_lr, h_batch, h_buffer, h_gamma, h_learn_starts, h_target_update, h_expl_frac, h_expl_final, h_grad, hyperparams_status]) 
                                          info="Limite la taille des gradients")
                
                h_vf = gr.Slider(0.3, 0.7, step=0.1, label="Value Function Coef", 
                                info="Poids de la value function")
                
                # Save button
                save_hyperparams_btn = gr.Button("💾 Save Hyperparameters", variant="primary")
                hyperparams_status = gr.Textbox(label="Status", interactive=False)
                
                back_hyperparams = gr.Button("← Back")
                
                # Load initial values
                params = load_hyperparams()
                h_nsteps.value = params["n_steps"]
                h_batch.value = params["batch_size"]
                h_gamma.value = params["gamma"]
                h_gae.value = params["gae_lambda"]
                h_ent.value = params["ent_coef"]
                h_lr.value = params["learning_rate"]
                h_clip.value = params["clip_range"]
                h_grad.value = params["max_grad_norm"]
                h_vf.value = params["vf_coef"]


            # OPTIMIZATION VIEW
            with gr.Column(visible=False) as optimization_view:
                gr.Markdown("## 🧪 Hyperparameter Optimization (Optuna)")
                gr.Markdown("""
                **C'est quoi ?**  
                Cet outil lance une recherche automatique pour trouver les **meilleurs hyperparamètres** (vitesse d'apprentissage, taille du batch, etc.) pour l'IA.
                
                **Comment ça marche ?**  
                Il va lancer plusieurs entraînements courts (Trials) en parallèle ou séquentiellement et comparer leurs performances.
                 gr.Markdown("Il sauvegarde dans `data/optimization/best_hyperparams.txt`.")
                """)
                
                with gr.Row():
                    n_trials_input = gr.Number(value=20, label="Nombre d'essais (Trials)", precision=0)
                    start_opt_btn = gr.Button("▶️ Launch New Optimization", variant="primary")
                
                opt_status = gr.Textbox(label="Status", interactive=False)
                opt_logs = gr.Textbox(label="📜 Recent Logs", lines=15, interactive=False)
                back_opt = gr.Button("← Back")
    
    # === HANDLERS ===
    
    def show_dashboard():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "dashboard", None
    
    def show_training():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "training", None
    
    def show_new_training():
        # Refresh model list for dropdown
        models = ["None (New Model)"] + [m['name'] for m in get_models()]
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                "new_training", None, gr.update(choices=models, value="None (New Model)"))
    
    def show_optimization():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                "optimization", None)
    
    def show_tuning(pid):
        defaults = load_rewards(pid)
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "tuning", pid, 
                defaults['progress_weight'], defaults['track_penalty'], defaults['speed_weight'], 
                defaults['wall_penalty'], defaults['checkpoint_bonus'], defaults['time_bonus'], defaults['alignment_weight'],
                defaults['checkpoint_scaling'], defaults['ghost_following_weight'])

    dashboard_btn.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    new_train_btn.click(show_new_training, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid, base_model_input])
    opt_view_btn.click(show_optimization, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    tuning_btn.click(show_tuning, inputs=[selected_pid], outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid,
                                           r_progress, r_track_penalty, r_speed, r_wall_penalty, r_cp_bonus, r_time_bonus, r_align, r_scaling, r_ghost_follow])
    
    cancel_btn.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    back_train.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    back_proc.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    back_opt.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    back_tuning.click(show_dashboard, outputs=[dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])

    save_tuning_btn.click(save_rewards, [r_progress, r_track_penalty, r_speed, r_wall_penalty, r_cp_bonus, r_time_bonus, r_align, r_scaling, r_ghost_follow, selected_pid], [tuning_status])
    
    def start_and_show_training(steps, name, base, use_opt, visual, procedural):
        status = start_training(steps, name, base, use_opt, visual, procedural)
        # Return status + switch to training view
        return status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "training", None

    start_btn.click(start_and_show_training, [steps_slider, custom_name_input, base_model_input, use_opt_checkbox, visual_mode_checkbox, procedural_checkbox], [train_status, dashboard_view, training_view, process_view, new_training_view, optimization_view, tuning_view, current_view, selected_pid])
    
    start_opt_btn.click(start_optimization, [n_trials_input], [opt_status])
    play_manual_btn.click(start_game, outputs=[action_status])
    play_ai_btn.click(start_ai_with_model, [model_select, chk_play_procedural], [action_status])
    stop_all_btn.click(stop_all, outputs=[action_status])
    
    def do_delete(name):
        st, df = delete_model(name)
        return st, df, gr.update(choices=[m['name'] for m in get_models()])
    
    del_model_btn.click(do_delete, [model_select], [action_status, models_table, model_select])
    
    def refresh():
        models = get_models()
        df = load_monitor_data()
        has_data = isinstance(df, pd.DataFrame) and len(df) > 0
        n_eps = len(df) if has_data else 0
        best = round(df['r'].max(), 1) if has_data else 0
        return (len(models), n_eps, best,
                len(get_running_processes()), get_models_df(), 
                gr.update(choices=[m['name'] for m in models]))
    
    refresh_btn.click(refresh, outputs=[n_models, n_eps, best_rew, n_procs, models_table, model_select])
    
    # === AUTO REFRESH ===
    timer = gr.Timer(2)
    last_proc_list = gr.State([])
    
    def auto_refresh(view, pid, refresh_count, last_procs):
        cpu, ram = get_system_stats()
        
        # Default metrics
        m = {"fps": 0, "steps": 0, "reward": 0.0}
        prog = 0
        log_txt = "Waiting..."
        run_id = None
        
        # Get detailed info if training
        if pid and view == "training":
             info = get_training_info(pid)
             prog = info['progress']
             m['fps'] = info['fps']
             m['steps'] = info['current']
             run_id = info.get('run_id')
             
             log = get_latest_log("training")
             raw_log = read_log(log, 20) if log else ""
             
             m2 = parse_metrics(raw_log)
             m['reward'] = m2['reward']
             
             spm = info['fps'] * 60
             spm_str = f"{spm:,}".replace(",", " ")
             
             log_txt = f"=== 📊 STATUS ===\nStep: {info['current']} / {info['total']}\nProg: {info['progress']}%\nFPS:  {info['fps']} (🚀 {spm_str} steps/min)\nETA:  {info['eta']}\n\n=== 📜 LOGS ===\n{raw_log}"
             
        elif view == "dashboard":
             log = get_latest_log("training")
             log_txt = read_log(log, 20) if log else "No active training log"
             m = parse_metrics(log_txt)
             prog = get_training_progress()

        # Advanced stats from CSV
        avg_cp, success, lap, best, max_cp = get_advanced_stats(run_id)
        
        # Charts Data Preparation
        full_df = get_chart_data(run_id)
        
        # Split into view-specific dataframes
        # This allows each plot to only show relevant lines without clutter
        df_reward = full_df[full_df['Metric'] == 'Reward']
        df_success = full_df[full_df['Metric'].isin(['Success Rate (%)', 'Checkpoints'])]
        df_length = full_df[full_df['Metric'] == 'Length']
        
        # Process List Logic (Smart Refresh)
        procs = get_running_processes()
        current_proc_signature = sorted([p['pid'] for p in procs])
        last_proc_signature = sorted([p['pid'] for p in last_procs])
        
        new_refresh_count = refresh_count
        new_last_procs = last_procs
        
        if current_proc_signature != last_proc_signature:
            new_refresh_count += 1
            new_last_procs = procs
        
        # Process Info Logic
        proc_info = ""
        proc_log = ""
        if pid:
            p = next((x for x in procs if x['pid'] == pid), None)
            if p:
                proc_info = f"**{p['label']}** | PID {pid} | Running: {format_time(p['start_time'])}"
                if p['type'] == 'ai':
                    ai_log = get_latest_log("ai")
                    proc_log = read_log(ai_log, 25) if ai_log else "No log"
                elif p['type'] == 'game':
                    game_log = get_latest_log("game")
                    proc_log = read_log(game_log, 25) if game_log else "No log"
        
        # Optimization logs
        opt_log_txt = ""
        if view == "optimization":
            o_log = get_latest_log("optimization")
            opt_log_txt = read_log(o_log, 30) if o_log else "Ready to start..."
        
        return (f"{cpu:.0f}%", f"{ram:.0f}%",
                m['fps'], m['steps'], m['reward'], prog, log_txt, 
                df_reward, df_success, df_length, # 3 Dataframes
                avg_cp, max_cp, success, lap, best,
                proc_info, proc_log,
                opt_log_txt,
                new_refresh_count,
                new_last_procs)
    
    timer.tick(auto_refresh, [current_view, selected_pid, process_refresh, last_proc_list],
               [cpu_txt, ram_txt,
                t_fps, t_steps, t_rew, t_prog, t_logs, 
                plot_reward, plot_success, plot_length, # 3 Plots
                t_cps, t_max_cp, t_success, t_lap, t_best,
                p_info, p_logs,
                opt_logs,
                process_refresh,
                last_proc_list])

# Apply CSS
app.css = CUSTOM_CSS

if __name__ == "__main__":
    print("🏎️ Starting AI Racing Lab...")
    app.launch(server_name="127.0.0.1")
