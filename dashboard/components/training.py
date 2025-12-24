"""
Training tab component - Start and monitor training.
"""
import gradio as gr
from dashboard.utils.config import TRAINING_PRESETS, DEFAULT_TRAINING_STEPS
from dashboard.utils.processes import launch_training, get_running_processes
from dashboard.utils.data import read_log, parse_training_metrics, load_monitor_data
import glob
import os


def create_training_tab():
    """Create the training tab."""
    
    with gr.Tab("🧠 Training"):
        gr.Markdown("## Neural Network Training")
        
        with gr.Row():
            # Left: Training Controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # Presets
                preset_dropdown = gr.Dropdown(
                    choices=list(TRAINING_PRESETS.keys()),
                    value="Standard",
                    label="Preset",
                    interactive=True
                )
                
                preset_info = gr.Markdown(f"*{TRAINING_PRESETS['Standard']['description']}*")
                
                # Custom settings
                with gr.Accordion("Advanced Settings", open=False):
                    steps_input = gr.Slider(
                        minimum=100_000,
                        maximum=5_000_000,
                        value=500_000,
                        step=100_000,
                        label="Training Steps",
                        interactive=True
                    )
                    
                    lr_input = gr.Slider(
                        minimum=1e-5,
                        maximum=1e-3,
                        value=3e-4,
                        label="Learning Rate",
                        interactive=True
                    )
                    
                    visual_mode = gr.Checkbox(
                        label="Visual Mode (slower, shows game)",
                        value=False
                    )
                
                gr.Markdown("---")
                
                # Launch buttons
                start_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", interactive=False, elem_classes=["panel-card"])
                
                # Preset change handler
                def update_preset(preset_name):
                    preset = TRAINING_PRESETS[preset_name]
                    return (
                        f"*{preset['description']}*",
                        preset['steps'],
                        preset['lr']
                    )
                
                preset_dropdown.change(
                    fn=update_preset,
                    inputs=[preset_dropdown],
                    outputs=[preset_info, steps_input, lr_input]
                )
            
            # Right: Live Monitoring
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live Progress")
                
                # Metrics
                with gr.Row():
                    fps_metric = gr.Number(value=0, label="⚡ FPS", interactive=False, elem_classes=["stat-card"])
                    steps_metric = gr.Number(value=0, label="👣 Steps", interactive=False, elem_classes=["stat-card"])
                    reward_metric = gr.Number(
                        value=0,
                        label="💰 Reward",
                        interactive=False,
                        precision=1,
                        elem_classes=["stat-card"]
                    )
                    progress_metric = gr.Number(
                        value=0,
                        label="📈 Progress %",
                        interactive=False,
                        precision=1,
                        elem_classes=["stat-card"]
                    )
                
                # Progress bar simulation
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Training Progress",
                    interactive=False,
                    elem_classes=["panel-card"]
                )
                
                # Logs
                gr.Markdown("### 📜 Training Logs")
                logs_display = gr.Textbox(
                    value="Waiting for training to start...",
                    label="",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                    elem_classes=["panel-card", "mono"]
                )
        
        # Event handlers
        def start_training_handler(steps, lr, visual):
            try:
                result = launch_training(int(steps), float(lr), visual)
                return f"✅ Training started! PID: {result['pid']}, Target: {int(steps):,} steps"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        start_btn.click(
            fn=start_training_handler,
            inputs=[steps_input, lr_input, visual_mode],
            outputs=[status_output]
        )
        
        # Refresh function
        def refresh_training():
            # Find latest training log
            logs = sorted(glob.glob("training_*.log"), key=os.path.getmtime, reverse=True)
            
            if not logs:
                return 0, 0, 0, 0, 0, "No training running..."
            
            log_content = read_log(logs[0], lines=30)
            metrics = parse_training_metrics(log_content)
            
            # Calculate progress (if we can find target steps)
            progress = 0
            procs = get_running_processes()
            training_procs = [p for p in procs if p['type'] == 'training']
            if training_procs:
                try:
                    cmd = training_procs[0]['cmd']
                    if '--steps' in cmd:
                        parts = cmd.split()
                        idx = parts.index('--steps')
                        target = int(parts[idx + 1])
                        progress = min(100, metrics['timesteps'] / target * 100)
                except:
                    pass
            
            return (
                metrics['fps'],
                metrics['timesteps'],
                metrics['reward'],
                progress,
                progress,
                read_log(logs[0], lines=25)
            )
        
        return {
            "refresh_fn": refresh_training,
            "outputs": [fps_metric, steps_metric, reward_metric, progress_metric, progress_bar, logs_display]
        }
