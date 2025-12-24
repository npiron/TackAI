"""
Playground tab component - Play and watch AI.
"""
import gradio as gr
from dashboard.utils.data import get_models
from dashboard.utils.processes import launch_ai_play, launch_manual_play, stop_all_processes


def create_playground_tab():
    """Create the playground tab."""
    
    with gr.Tab("🤖 Playground"):
        gr.Markdown("## Play & Watch")
        
        with gr.Row():
            # Watch AI
            with gr.Column():
                gr.Markdown("### 🤖 Watch AI Play")
                gr.Markdown("*Select a trained model and watch the AI drive!*")
                
                models = get_models()
                model_choices = [m['name'] for m in models] if models else []
                
                model_select = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Select Model",
                    interactive=True
                )
                
                if models:
                    # Show model info
                    selected_model = models[0]
                    model_info = gr.Markdown(
                        f"**Steps:** {selected_model['steps']:,} | **Size:** {selected_model['size_mb']:.1f} MB"
                    )
                
                watch_btn = gr.Button("▶️ Watch AI Drive", variant="primary", size="lg")
                watch_status = gr.Textbox(label="Status", interactive=False, elem_classes=["panel-card"])
                
                def watch_ai(model_name):
                    if not model_name:
                        return "❌ No model selected"
                    
                    models = get_models()
                    model = next((m for m in models if m['name'] == model_name), None)
                    
                    if not model:
                        return "❌ Model not found"
                    
                    try:
                        result = launch_ai_play(model['path'])
                        return f"✅ AI started! PID: {result['pid']} - Check game window"
                    except Exception as e:
                        return f"❌ Error: {str(e)}"
                
                watch_btn.click(
                    fn=watch_ai,
                    inputs=[model_select],
                    outputs=[watch_status]
                )
            
            # Manual Play
            with gr.Column():
                gr.Markdown("### 🎮 Play Manually")
                gr.Markdown("*Drive the car yourself!*")
                
                gr.Markdown("""
                **Controls:**
                - ↑ Arrow: Accelerate
                - ↓ Arrow: Brake
                - ← → : Steer
                - R: Reset
                - ESC: Quit
                """)
                
                play_btn = gr.Button("🎮 Start Game", variant="primary", size="lg")
                play_status = gr.Textbox(label="Status", interactive=False, elem_classes=["panel-card"])
                
                def start_manual():
                    try:
                        result = launch_manual_play()
                        return f"✅ Game launched! PID: {result['pid']} - Check your Dock"
                    except Exception as e:
                        return f"❌ Error: {str(e)}"
                
                play_btn.click(
                    fn=start_manual,
                    outputs=[play_status]
                )
        
        gr.Markdown("---")
        
        # Stop All
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⏹️ Stop Everything")
                stop_btn = gr.Button("🛑 Stop All Processes", variant="stop")
                stop_status = gr.Textbox(label="", interactive=False, elem_classes=["panel-card"])
                
                def stop_all():
                    count = stop_all_processes()
                    return f"⏹️ Stopped {count} process(es)"
                
                stop_btn.click(
                    fn=stop_all,
                    outputs=[stop_status]
                )
