# src/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config
from .inference import HousePricePredictor
from .utils import load_pickle

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    .feature-input {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata with caching."""
    try:
        config = Config()
        
        # Load model metadata
        metadata_path = config.MODELS_DIR / "model_metadata.pkl"
        metadata = load_pickle(metadata_path) if metadata_path.exists() else {}
        
        # Load evaluation report
        eval_path = config.DOCS_DIR / "evaluation_report.json"
        eval_data = {}
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
        
        return metadata, eval_data
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return {}, {}

def get_top_features():
    """Define the most important features for the UI."""
    return {
        'OverallQual': {
            'type': 'selectbox',
            'label': 'Overall Quality',
            'options': list(range(1, 11)),
            'help': 'Rates the overall material and finish of the house (1-10)',
            'default': 5
        },
        'GrLivArea': {
            'type': 'number_input',
            'label': 'Above Ground Living Area (sq ft)',
            'min_value': 300,
            'max_value': 5000,
            'value': 1500,
            'help': 'Above ground living area in square feet'
        },
        'TotalBsmtSF': {
            'type': 'number_input',
            'label': 'Total Basement Area (sq ft)',
            'min_value': 0,
            'max_value': 3000,
            'value': 1000,
            'help': 'Total basement area in square feet'
        },
        'GarageCars': {
            'type': 'selectbox',
            'label': 'Garage Cars Capacity',
            'options': [0, 1, 2, 3, 4],
            'help': 'Size of garage in car capacity',
            'default': 2
        },
        'YearBuilt': {
            'type': 'number_input',
            'label': 'Year Built',
            'min_value': 1870,
            'max_value': 2024,
            'value': 2000,
            'help': 'Original construction date'
        },
        'FullBath': {
            'type': 'selectbox',
            'label': 'Full Bathrooms',
            'options': [0, 1, 2, 3, 4],
            'help': 'Number of full bathrooms',
            'default': 2
        },
        'KitchenQual': {
            'type': 'selectbox',
            'label': 'Kitchen Quality',
            'options': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            'help': 'Kitchen quality rating',
            'default': 'TA'
        },
        'Neighborhood': {
            'type': 'selectbox',
            'label': 'Neighborhood',
            'options': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
                       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 
                       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 
                       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 
                       'Blueste'],
            'help': 'Physical locations within Ames city limits',
            'default': 'NAmes'
        }
    }

def create_input_form():
    """Create the input form for house features."""
    st.markdown("### 🏠 House Features")
    
    features = get_top_features()
    user_input = {}
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (feature, config) in enumerate(features.items()):
            if i % 2 == 0:  # First column
                if config['type'] == 'selectbox':
                    user_input[feature] = st.selectbox(
                        config['label'],
                        options=config['options'],
                        index=config['options'].index(config['default']),
                        help=config['help']
                    )
                elif config['type'] == 'number_input':
                    user_input[feature] = st.number_input(
                        config['label'],
                        min_value=config['min_value'],
                        max_value=config['max_value'],
                        value=config['value'],
                        help=config['help']
                    )
    
    with col2:
        for i, (feature, config) in enumerate(features.items()):
            if i % 2 == 1:  # Second column
                if config['type'] == 'selectbox':
                    user_input[feature] = st.selectbox(
                        config['label'],
                        options=config['options'],
                        index=config['options'].index(config['default']),
                        help=config['help']
                    )
                elif config['type'] == 'number_input':
                    user_input[feature] = st.number_input(
                        config['label'],
                        min_value=config['min_value'],
                        max_value=config['max_value'],
                        value=config['value'],
                        help=config['help']
                    )
    
    return user_input

def display_model_info(metadata, eval_data):
    """Display model information and performance metrics."""
    st.sidebar.markdown("## 📊 Model Information")
    
    # Model metadata
    if metadata:
        st.sidebar.markdown(f"**Model Type:** {metadata.get('model_type', 'Unknown')}")
        st.sidebar.markdown(f"**Best Score (R²):** {metadata.get('best_score', 0):.4f}")
    
    # Evaluation metrics
    if eval_data and 'model_evaluation_summary' in eval_data:
        metrics = eval_data['model_evaluation_summary']['metrics']
        performance = eval_data['performance_analysis']
        
        st.sidebar.markdown("### Performance Metrics")
        st.sidebar.markdown(f"**R² Score:** {metrics['r2']:.4f}")
        st.sidebar.markdown(f"**RMSE:** ${metrics['rmse']:,.0f}")
        st.sidebar.markdown(f"**MAE:** ${metrics['mae']:,.0f}")
        st.sidebar.markdown(f"**MAPE:** {metrics['mape']:.1f}%")
        
        # Performance rating
        st.sidebar.markdown(f"**Overall Rating:** {performance['overall_performance']}")
        
        # Test samples
        st.sidebar.markdown(f"**Test Samples:** {eval_data['model_evaluation_summary']['test_samples']}")

def create_performance_chart(eval_data):
    """Create a performance visualization."""
    if not eval_data or 'model_evaluation_summary' not in eval_data:
        return
    
    metrics = eval_data['model_evaluation_summary']['metrics']
    
    # Create subplot for metrics
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Error Metrics', 'Model Performance'),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Error metrics bar chart
    error_metrics = ['MAE', 'RMSE']
    error_values = [metrics['mae'], metrics['rmse']]
    
    fig.add_trace(
        go.Bar(x=error_metrics, y=error_values, name='Error Metrics'),
        row=1, col=1
    )
    
    # R² gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics['r2'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "R² Score"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.6], 'color': "lightgray"},
                            {'range': [0.6, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict house prices using machine learning")
    
    # Load model and metadata
    metadata, eval_data = load_model_and_metadata()
    
    # Sidebar with model info
    display_model_info(metadata, eval_data)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        user_input = create_input_form()
        
        # Prediction button
        if st.button("🚀 Predict House Price", type="primary", use_container_width=True):
            try:
                # Initialize predictor
                predictor = HousePricePredictor(model_source="local")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = predictor.predict(user_input)
                
                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>Predicted House Price</h2>
                    <h1>${prediction:,.2f}</h1>
                    <p>Based on the provided features</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Save prediction to docs
                prediction_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input_features": user_input,
                    "predicted_price": float(prediction),
                    "predicted_price_formatted": f"${prediction:,.2f}"
                }
                
                config = Config()
                predictions_path = config.DOCS_DIR / "streamlit_predictions.json"
                
                # Load existing predictions or create new list
                try:
                    with open(predictions_path, 'r') as f:
                        all_predictions = json.load(f)
                except FileNotFoundError:
                    all_predictions = []
                
                all_predictions.append(prediction_data)
                
                # Save updated predictions
                with open(predictions_path, 'w') as f:
                    json.dump(all_predictions, f, indent=2)
                
                st.success("✅ Prediction completed and saved!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    with col2:
        # Performance visualization
        if eval_data:
            st.markdown("### 📈 Model Performance")
            fig = create_performance_chart(eval_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.markdown("### 🔍 Model Details")
        if metadata:
            st.markdown(f"**Algorithm:** {metadata.get('model_type', 'Unknown')}")
            st.markdown(f"**Training Date:** {metadata.get('timestamp', 'Unknown')}")
        
        if eval_data and 'performance_analysis' in eval_data:
            performance = eval_data['performance_analysis']
            st.markdown(f"**R² Interpretation:** {performance['r2_interpretation']}")
            st.markdown(f"**RMSE Interpretation:** {performance['rmse_interpretation']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p>House Price Prediction MLOps Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 