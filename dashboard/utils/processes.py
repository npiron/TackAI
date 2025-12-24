"""
Dashboard utilities for process management.
"""
import os
import sys
import time
import signal
import psutil
import subprocess


def get_python_cmd():
    """Get the Python executable path (venv or system)."""
    venv_python = os.path.abspath(".venv/bin/python")
    return venv_python if os.path.exists(venv_python) else sys.executable


def get_running_processes():
    """Get all running Trackmania-related processes."""
    procs = []
    keywords = {
        "rl_train.py": ("🧠 Training", "training"),
        "rl_play.py": ("🤖 AI Play", "ai_play"),
        "trackmania_clone.py": ("🎮 Manual", "manual"),
        "rl_optimize.py": ("🔧 Optimizer", "optimizer"),
    }
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and len(cmdline) > 1 and "python" in proc.info['name'].lower():
                for arg in cmdline:
                    for kw, (label, ptype) in keywords.items():
                        if kw in arg:
                            procs.append({
                                "pid": proc.info['pid'],
                                "label": label,
                                "type": ptype,
                                "script": kw,
                                "cmd": " ".join(cmdline),
                                "started": time.strftime('%H:%M:%S', time.localtime(proc.info['create_time'])),
                                "create_time": proc.info['create_time']
                            })
                            break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return sorted(procs, key=lambda x: x['create_time'], reverse=True)


def stop_process(pid):
    """Stop a process by PID."""
    try:
        os.kill(pid, signal.SIGINT)
        time.sleep(0.5)
        return True
    except:
        return False


def stop_all_processes():
    """Stop all Trackmania-related processes."""
    procs = get_running_processes()
    for p in procs:
        stop_process(p['pid'])
    time.sleep(1)
    return len(procs)


def launch_training(steps: int, lr: float = 3e-4, visual: bool = False):
    """Launch a training session."""
    run_id = str(int(time.time()))
    log_file = f"training_{run_id}.log"
    
    cmd = [get_python_cmd(), "-u", "rl_train.py", 
           "--steps", str(steps), 
           "--lr", str(lr),
           "--run-id", run_id]
    
    if visual:
        cmd.append("--visual")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    return {
        "pid": proc.pid,
        "run_id": run_id,
        "log_file": log_file,
        "steps": steps
    }


def launch_ai_play(model_path: str):
    """Launch AI replay with a specific model."""
    import shutil
    shutil.copy(model_path, "ppo_timeattack.zip")
    
    log_file = f"ai_{int(time.time())}.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen([get_python_cmd(), "rl_play.py"], 
                               stdout=f, stderr=subprocess.STDOUT)
    
    return {"pid": proc.pid, "log_file": log_file}


def launch_manual_play():
    """Launch manual game."""
    log_file = f"game_{int(time.time())}.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen([get_python_cmd(), "trackmania_clone.py"],
                               stdout=f, stderr=subprocess.STDOUT)
    
    return {"pid": proc.pid, "log_file": log_file}


def get_system_stats():
    """Get CPU and memory usage."""
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "ram_used_gb": mem.used / (1024**3),
        "ram_total_gb": mem.total / (1024**3)
    }
