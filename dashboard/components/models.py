"""
Models tab component - Model management.
"""
import gradio as gr
import pandas as pd
from dashboard.utils.data import get_models, get_models_dataframe, delete_model


def create_models_tab():
    """Create the models management tab."""
    
    with gr.Tab("📦 Models"):
        gr.Markdown("## Model Management")
        
        # Models Table
        gr.Markdown("### 📋 Trained Models")
        
        models_table = gr.Dataframe(
            value=get_models_dataframe(),
            headers=["Model", "Steps", "Size", "Created"],
            interactive=False,
            wrap=True,
            elem_classes=["panel-card"]
        )
        
        # Refresh button
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        
        def refresh_models():
            return get_models_dataframe()
        
        refresh_btn.click(
            fn=refresh_models,
            outputs=[models_table]
        )
        
        gr.Markdown("---")
        
        # Model Actions
        gr.Markdown("### ⚡ Actions")
        
        models = get_models()
        model_choices = [m['name'] for m in models] if models else []
        
        with gr.Row():
            action_model = gr.Dropdown(
                choices=model_choices,
                label="Select Model",
                interactive=True
            )
            
            download_btn = gr.Button("⬇️ Download")
            delete_btn = gr.Button("🗑️ Delete", variant="stop")
        
        action_status = gr.Textbox(label="Status", interactive=False, elem_classes=["panel-card"])
        
        # Download handler
        def get_model_file(model_name):
            if not model_name:
                return None
            models = get_models()
            model = next((m for m in models if m['name'] == model_name), None)
            if model:
                return model['path']
            return None
        
        download_file = gr.File(label="Download", visible=False)
        
        # Delete handler
        def delete_handler(model_name):
            if not model_name:
                return "❌ No model selected", get_models_dataframe()
            
            if delete_model(model_name):
                return f"✅ Deleted: {model_name}", get_models_dataframe()
            return f"❌ Failed to delete: {model_name}", get_models_dataframe()
        
        delete_btn.click(
            fn=delete_handler,
            inputs=[action_model],
            outputs=[action_status, models_table]
        )
        
        gr.Markdown("---")
        
        # Storage Info
        gr.Markdown("### 💾 Storage")
        
        models = get_models()
        total_size = sum(m['size_mb'] for m in models)
        
        gr.Markdown(f"""
        - **Total Models:** {len(models)}
        - **Total Size:** {total_size:.1f} MB
        - **Location:** `logs/*.zip`
        """)
