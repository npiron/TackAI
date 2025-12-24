"""
Overview tab component - Quick stats and active processes.
"""
import gradio as gr
from dashboard.utils.data import get_models, get_training_stats, load_monitor_data
from dashboard.utils.processes import get_running_processes, get_system_stats


def create_overview_tab():
    """Create the overview/home tab."""
    
    with gr.Tab("🏠 Overview"):
        gr.Markdown("## Dashboard Overview")
        
        # Quick Stats
        with gr.Row():
            with gr.Column(scale=1):
                models_count = gr.Number(
                    value=len(get_models()),
                    label="🏷️ Models Trained",
                    interactive=False,
                    elem_classes=["stat-card"]
                )
            with gr.Column(scale=1):
                stats = get_training_stats()
                best_reward = gr.Number(
                    value=stats['best_reward'],
                    label="🏆 Best Reward",
                    interactive=False,
                    precision=1,
                    elem_classes=["stat-card"]
                )
            with gr.Column(scale=1):
                total_episodes = gr.Number(
                    value=stats['total_episodes'],
                    label="📊 Total Episodes",
                    interactive=False,
                    elem_classes=["stat-card"]
                )
            with gr.Column(scale=1):
                procs = get_running_processes()
                status_text = f"🟢 {len(procs)} Running" if procs else "⚪ Idle"
                status = gr.Textbox(
                    value=status_text,
                    label="Status",
                    interactive=False,
                    elem_classes=["stat-card"]
                )
        
        gr.Markdown("---")
        
        # Active Processes
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🔄 Active Processes")
                
                def get_processes_text():
                    procs = get_running_processes()
                    if not procs:
                        return "No active processes"
                    return "\n".join([
                        f"{p['label']} (PID {p['pid']}) - Started {p['started']}"
                        for p in procs
                    ])
                
                processes_display = gr.Textbox(
                    value=get_processes_text(),
                    label="",
                    lines=4,
                    interactive=False,
                    elem_classes=["panel-card", "mono"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 💻 System")
                sys_stats = get_system_stats()
                cpu_display = gr.Textbox(
                    value=f"CPU: {sys_stats['cpu_percent']:.0f}%",
                    label="",
                    interactive=False,
                    elem_classes=["panel-card"]
                )
                ram_display = gr.Textbox(
                    value=f"RAM: {sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.0f} GB ({sys_stats['ram_percent']:.0f}%)",
                    label="",
                    interactive=False,
                    elem_classes=["panel-card"]
                )
        
        gr.Markdown("---")
        
        # Recent Learning Curve
        gr.Markdown("### 📈 Recent Training Progress")
        
        df = load_monitor_data()
        if df is not None and len(df) > 2:
            import pandas as pd
            plot_df = df[['r']].tail(200).copy()
            if len(plot_df) >= 10:
                window = min(20, len(plot_df) // 2)
                plot_df['smooth'] = plot_df['r'].rolling(window=window).mean()
            
            chart = gr.LinePlot(
                value=plot_df.reset_index(),
                x="index",
                y="r",
                height=250,
                width=800,
            )
        else:
            gr.Markdown("*No training data yet. Start training to see progress.*")
        
        # Return components for refresh
        return {
            "models_count": models_count,
            "best_reward": best_reward,
            "total_episodes": total_episodes,
            "status": status,
            "processes": processes_display,
            "cpu": cpu_display,
            "ram": ram_display,
        }
