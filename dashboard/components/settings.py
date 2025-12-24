"""
Settings tab component - Configuration and tuning.
"""
import gradio as gr


def create_settings_tab():
    """Create the settings tab."""
    
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("## Configuration")
        
        with gr.Row():
            # Physics Settings
            with gr.Column():
                gr.Markdown("### 🏎️ Physics")
                gr.Markdown("*Adjust car physics parameters*")
                
                lateral_grip = gr.Slider(
                    minimum=0.05, maximum=0.5, value=0.15, step=0.01,
                    label="Lateral Grip",
                    info="Lower = more drift, Higher = more grip"
                )
                
                turn_speed = gr.Slider(
                    minimum=0.02, maximum=0.15, value=0.08, step=0.01,
                    label="Turn Speed",
                    info="How fast the car turns"
                )
                
                max_speed = gr.Slider(
                    minimum=10, maximum=30, value=20, step=1,
                    label="Max Speed",
                    info="Maximum car speed"
                )
                
                friction = gr.Slider(
                    minimum=0.9, maximum=0.99, value=0.98, step=0.01,
                    label="Friction",
                    info="Speed decay per frame"
                )
                
                save_physics_btn = gr.Button("💾 Save Physics", variant="primary")
                physics_status = gr.Textbox(label="", interactive=False, visible=False)
            
            # Reward Settings
            with gr.Column():
                gr.Markdown("### 🎯 Reward Tuning")
                gr.Markdown("*Adjust reward function weights*")
                
                checkpoint_bonus = gr.Slider(
                    minimum=10, maximum=200, value=100, step=10,
                    label="Checkpoint Bonus",
                    info="Reward for reaching a checkpoint"
                )
                
                progress_multiplier = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Progress Multiplier",
                    info="Reward for moving towards checkpoint"
                )
                
                off_track_penalty = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                    label="Off-Track Penalty",
                    info="Penalty per frame off-road"
                )
                
                wall_penalty = gr.Slider(
                    minimum=1, maximum=10, value=3, step=0.5,
                    label="Wall Collision Penalty",
                    info="Penalty for hitting walls"
                )
                
                save_rewards_btn = gr.Button("💾 Save Rewards", variant="primary")
        
        gr.Markdown("---")
        
        # Track Settings
        gr.Markdown("### 🛣️ Track")
        
        with gr.Row():
            with gr.Column():
                num_checkpoints = gr.Number(value=10, label="Number of Checkpoints", interactive=False)
                checkpoint_radius = gr.Number(value=60, label="Checkpoint Radius", interactive=False)
            with gr.Column():
                finish_radius = gr.Number(value=50, label="Finish Radius", interactive=False)
                track_width = gr.Number(value=120, label="Track Width", interactive=False)
        
        gr.Markdown("*Track settings require code modification*")
        
        gr.Markdown("---")
        
        # Maintenance
        gr.Markdown("### 🧹 Maintenance")
        
        with gr.Row():
            clear_logs_btn = gr.Button("🗑️ Clear Logs")
            clear_cache_btn = gr.Button("🧹 Clear Cache")
        
        maintenance_status = gr.Textbox(label="Status", interactive=False)
        
        def clear_logs():
            from dashboard.utils.data import clear_logs
            clear_logs()
            return "✅ Logs cleared"
        
        clear_logs_btn.click(
            fn=clear_logs,
            outputs=[maintenance_status]
        )
