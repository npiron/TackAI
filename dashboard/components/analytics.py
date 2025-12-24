"""
Analytics tab component - Charts and statistics.
"""
import gradio as gr
import pandas as pd
from dashboard.utils.data import load_monitor_data, get_training_stats, get_models


def create_analytics_tab():
    """Create the analytics tab."""
    
    with gr.Tab("📊 Analytics"):
        gr.Markdown("## Training Analytics")
        
        # Summary Stats
        with gr.Row():
            stats = get_training_stats()
            
            with gr.Column():
                gr.Number(value=stats['total_episodes'], label="Total Episodes", interactive=False, elem_classes=["stat-card"])
            with gr.Column():
                gr.Number(value=round(stats['best_reward'], 1), label="Best Reward", interactive=False, elem_classes=["stat-card"])
            with gr.Column():
                gr.Number(value=round(stats['avg_reward'], 1), label="Average Reward", interactive=False, elem_classes=["stat-card"])
            with gr.Column():
                gr.Number(
                    value=round(stats['recent_reward'], 1),
                    label="Recent Avg (10)",
                    interactive=False,
                    elem_classes=["stat-card"]
                )
        
        gr.Markdown("---")
        
        # Learning Curve
        gr.Markdown("### 📈 Learning Curve")
        
        df = load_monitor_data()
        
        if df is not None and len(df) > 5:
            # Prepare data
            plot_df = df[['r']].copy().reset_index()
            plot_df.columns = ['Episode', 'Reward']
            
            # Add moving average
            window = min(50, len(plot_df) // 4) if len(plot_df) > 20 else 5
            plot_df['MA'] = plot_df['Reward'].rolling(window=window, min_periods=1).mean()
            
            # Main chart
            learning_chart = gr.LinePlot(
                value=plot_df,
                x="Episode",
                y="Reward",
                height=300,
                width=900,
                title="Episode Rewards Over Time"
            )
            
            gr.Markdown(f"*Showing {len(plot_df)} episodes. Green: Raw, Blue: Moving Average (window={window})*")
            
            gr.Markdown("---")
            
            # Distribution
            gr.Markdown("### 📊 Reward Distribution")
            
            with gr.Row():
                with gr.Column():
                    # Histogram data
                    hist_data = plot_df['Reward'].describe()
                    
                    gr.Markdown(f"""
                    **Statistics:**
                    - Min: {hist_data['min']:.1f}
                    - Max: {hist_data['max']:.1f}
                    - Mean: {hist_data['mean']:.1f}
                    - Std: {hist_data['std']:.1f}
                    - Median: {hist_data['50%']:.1f}
                    """)
                
                with gr.Column():
                    # Recent trend
                    if len(plot_df) >= 100:
                        early = plot_df['Reward'].head(50).mean()
                        late = plot_df['Reward'].tail(50).mean()
                        improvement = late - early
                        
                        gr.Markdown(f"""
                        **Learning Trend:**
                        - First 50 episodes avg: {early:.1f}
                        - Last 50 episodes avg: {late:.1f}
                        - Improvement: **{'+' if improvement > 0 else ''}{improvement:.1f}**
                        """)
                    else:
                        gr.Markdown("*Need 100+ episodes for trend analysis*")
        
        else:
            gr.Markdown("*No training data available. Start training to see analytics.*")
        
        gr.Markdown("---")
        
        # Model Comparison
        gr.Markdown("### 🏆 Model Leaderboard")
        
        models = get_models()
        if models:
            leaderboard_df = pd.DataFrame([{
                "Rank": i + 1,
                "Model": m['name'],
                "Steps": f"{m['steps']:,}",
                "Created": m['modified'].strftime("%m/%d %H:%M")
            } for i, m in enumerate(sorted(models, key=lambda x: x['steps'], reverse=True)[:10])])
            
            gr.Dataframe(
                value=leaderboard_df,
                headers=["Rank", "Model", "Steps", "Created"],
                interactive=False,
                elem_classes=["panel-card"]
            )
        else:
            gr.Markdown("*No models trained yet.*")
