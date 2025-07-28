"""
VIX Direction Prediction Model Showcase
Interactive Streamlit dashboard for comparing model results
"""

import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="VIX Direction Prediction Showcase",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_all_results():
    """Load all model results from the results directory"""
    results_dir = 'results'
    
    pattern = os.path.join(results_dir, 'complete_pipeline_*.json')
    result_files = glob.glob(pattern)
    
    all_results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config_name = data.get('config_used', 'unknown')
            version = data.get('version', 1)
            timestamp = data.get('timestamp', 'unknown')
            
            lstm_results = data.get('lstm_training', {})
            summary = lstm_results.get('summary', {})
            
            result_info = {
                'config_name': config_name,
                'version': version,
                'timestamp': timestamp,
                'file_path': file_path,
                'avg_accuracy': summary.get('avg_accuracy', 0),
                'std_accuracy': summary.get('std_accuracy', 0),
                'improvement_over_random': summary.get('improvement_over_random', 0),
                'success': summary.get('success', False),
                'individual_accuracies': summary.get('individual_accuracies', []),
                'full_data': data
            }
            
            all_results.append(result_info)
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    
    return sorted(all_results, key=lambda x: (x['config_name'], x['version']))

def create_performance_comparison(results):
    """Create performance comparison chart"""
    df = pd.DataFrame([
        {
            'Model': f"{r['config_name']}_v{r['version']}",
            'Config': r['config_name'],
            'Version': r['version'],
            'Accuracy': r['avg_accuracy'],
            'Std Dev': r['std_accuracy'],
            'Improvement': r['improvement_over_random'],
            'Success': r['success']
        }
        for r in results
    ])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Accuracy Comparison', 'Improvement over Random', 
                       'Accuracy vs Stability', 'Success Rate by Config'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(
            x=df['Model'],
            y=df['Accuracy'],
            error_y=dict(type='data', array=df['Std Dev']),
            name='Accuracy',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # random baseline
    fig.add_hline(y=0.333, line_dash="dash", line_color="red", 
                  annotation_text="Random Baseline (33.3%)", row=1, col=1)
    
    # Improvement over random
    colors = ['green' if imp > 0 else 'red' for imp in df['Improvement']]
    fig.add_trace(
        go.Bar(
            x=df['Model'],
            y=df['Improvement'],
            name='Improvement %',
            marker_color=colors
        ),
        row=1, col=2
    )
    
    # Accuracy vs Stability (scatter)
    fig.add_trace(
        go.Scatter(
            x=df['Std Dev'],
            y=df['Accuracy'],
            mode='markers+text',
            text=df['Model'],
            textposition="top center",
            name='Accuracy vs Stability',
            marker=dict(size=10, color=df['Improvement'], colorscale='RdYlGn')
        ),
        row=2, col=1
    )
    
    # Success rate per config
    success_by_config = df.groupby('Config')['Success'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=success_by_config['Config'],
            y=success_by_config['Success'],
            name='Success Rate',
            marker_color='gold'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Model Performance Overview")
    return fig

def display_model_details(result):
    """Display detailed information for a specific model"""
    lstm_data = result['full_data'].get('lstm_training', {})
    fold_results = lstm_data.get('fold_results', [])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Accuracy",
            value=f"{result['avg_accuracy']:.1%}",
            delta=f"{result['improvement_over_random']:.1f}% vs random"
        )
    
    with col2:
        st.metric(
            label="Stability (Std Dev)",
            value=f"{result['std_accuracy']:.3f}",
            delta="Lower is better"
        )
    
    with col3:
        st.metric(
            label="Success Status",
            value="✅ SUCCESS" if result['success'] else "❌ NEEDS WORK"
        )
    
    # Fold-by-fold results
    if fold_results:
        st.subheader("Fold-by-Fold Results")
        
        fold_df = pd.DataFrame([
            {
                'Fold': f"Fold {r['fold']}",
                'Accuracy': r['accuracy'],
                'Train Sequences': r['num_train_sequences'],
                'Test Sequences': r['num_test_sequences'],
                'Period': f"{r['window_info']['start_date'][:10]} to {r['window_info']['end_date'][:10]}"
            }
            for r in fold_results
        ])
        
        st.dataframe(fold_df, use_container_width=True)
        
        # Fold accuracy chart
        fig = px.bar(
            fold_df, 
            x='Fold', 
            y='Accuracy',
            title='Accuracy by Fold',
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0.333, line_dash="dash", line_color="red", annotation_text="Random Baseline")
        st.plotly_chart(fig, use_container_width=True)

def show_model_visualizations():
    """Show available model visualization images"""
    st.subheader("Model Visualizations")
    
    png_files = glob.glob('results/*.png')
    
    if not png_files:
        st.warning("No visualization files found")
        return
    
    model_pngs = {}
    for png_file in png_files:
        filename = os.path.basename(png_file)
        if 'lstm_training_' in filename:
            model_name = filename.replace('lstm_training_', '').replace('.png', '')
            model_pngs[model_name] = png_file
    
    if model_pngs:
        selected_model = st.selectbox("Select Model Visualization", list(model_pngs.keys()))
        
        if selected_model:
            st.image(model_pngs[selected_model], caption=f"Results for {selected_model}")

def create_summary_table(results):
    """Create a summary table of all models"""
    summary_data = []
    
    for r in results:
        summary_data.append({
            'Model': f"{r['config_name']}_v{r['version']}",
            'Config': r['config_name'],
            'Version': r['version'],
            'Accuracy': f"{r['avg_accuracy']:.1%}",
            'Std Dev': f"{r['std_accuracy']:.3f}",
            'Improvement': f"{r['improvement_over_random']:.1f}%",
            'Status': "✅ Success" if r['success'] else "❌ Needs Work",
            'Date': r['timestamp'].split(' ')[0] if ' ' in r['timestamp'] else r['timestamp']
        })
    
    return pd.DataFrame(summary_data)

def main():
    st.title("📈 VIX Direction Prediction Model Showcase")
    st.markdown("### Interactive Dashboard for Model Performance Analysis")
    
    results = load_all_results()
    
    if not results:
        st.error("No model results found in the results directory")
        return
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Comparison", "Individual Model Analysis", "Visualizations", "Raw Data"]
    )
    
    if page == "Overview":
        st.header("Model Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", len(results))
        
        with col2:
            successful_models = sum(1 for r in results if r['success'])
            st.metric("Successful Models", successful_models)
        
        with col3:
            best_accuracy = max(r['avg_accuracy'] for r in results)
            st.metric("Best Accuracy", f"{best_accuracy:.1%}")
        
        with col4:
            avg_improvement = np.mean([r['improvement_over_random'] for r in results])
            st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
        
        st.subheader("All Models Summary")
        summary_df = create_summary_table(results)
        st.dataframe(summary_df, use_container_width=True)
        
        st.subheader("Performance Comparison")
        comparison_fig = create_performance_comparison(results)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        model_options = [f"{r['config_name']}_v{r['version']}" for r in results]
        selected_models = st.multiselect("Select models to compare", model_options, default=model_options[:3])
        
        if selected_models:
            selected_results = [r for r in results if f"{r['config_name']}_v{r['version']}" in selected_models]
            
            comparison_data = []
            for r in selected_results:
                comparison_data.append({
                    'Model': f"{r['config_name']}_v{r['version']}",
                    'Accuracy': r['avg_accuracy'],
                    'Stability': r['std_accuracy'],
                    'Improvement': r['improvement_over_random'],
                    'Individual Accuracies': r['individual_accuracies']
                })
            
            for i, data in enumerate(comparison_data):
                with st.expander(f"📊 {data['Model']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{data['Accuracy']:.1%}")
                    with col2:
                        st.metric("Stability", f"{data['Stability']:.3f}")
                    with col3:
                        st.metric("Improvement", f"{data['Improvement']:.1f}%")
                    
                    # Individual fold accuracies
                    if data['Individual Accuracies']:
                        fold_data = pd.DataFrame({
                            'Fold': range(len(data['Individual Accuracies'])),
                            'Accuracy': data['Individual Accuracies']
                        })
                        fig = px.line(fold_data, x='Fold', y='Accuracy', title=f"{data['Model']} - Fold Accuracies")
                        fig.add_hline(y=0.333, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Individual Model Analysis":
        st.header("Individual Model Analysis")
        
        model_options = [f"{r['config_name']}_v{r['version']}" for r in results]
        selected_model = st.selectbox("Select a model for detailed analysis", model_options)
        
        if selected_model:
            selected_result = next(r for r in results if f"{r['config_name']}_v{r['version']}" == selected_model)
            
            st.subheader(f"Analysis: {selected_model}")
            display_model_details(selected_result)
            
            # Configuration details
            with st.expander("Configuration Details"):
                config_data = selected_result['full_data'].get('config', {})
                st.json(config_data)
    
    elif page == "Visualizations":
        show_model_visualizations()
    
    elif page == "Raw Data":
        st.header("Raw Data Explorer")
        
        model_options = [f"{r['config_name']}_v{r['version']}" for r in results]
        selected_model = st.selectbox("Select a model to view raw data", model_options)
        
        if selected_model:
            selected_result = next(r for r in results if f"{r['config_name']}_v{r['version']}" == selected_model)
            
            st.subheader(f"Raw Data: {selected_model}")
            
            st.json(selected_result['full_data'])

if __name__ == "__main__":
    main()