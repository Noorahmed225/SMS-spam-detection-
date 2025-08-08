"""
Streamlit Web Application for SMS Spam Detection
Provides a user-friendly web interface for spam detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from data_collection import get_dataset
from preprocessing import TextPreprocessor
from feature_extraction import prepare_features
from model_training import train_and_evaluate, SpamClassifier
from prediction import SpamPredictor

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .spam-prediction {
        border-left: 4px solid #ff4444;
    }
    .ham-prediction {
        border-left: 4px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Main function for the Streamlit app
    """
    # Header
    st.markdown('<h1 class="main-header">📱 SMS Spam Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🎯 Predict", "📊 Analysis", "🤖 Train Models", "📈 Results"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Predict":
        show_predict_page()
    elif page == "📊 Analysis":
        show_analysis_page()
    elif page == "🤖 Train Models":
        show_train_page()
    elif page == "📈 Results":
        show_results_page()

def show_home_page():
    """
    Display the home page
    """
    st.markdown("## Welcome to SMS Spam Detection System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Project
        
        This SMS Spam Detection system uses machine learning to classify text messages as either spam or legitimate (ham). 
        The system employs multiple algorithms including:
        
        - **Naive Bayes**: Fast and effective for text classification
        - **Logistic Regression**: Good baseline with interpretability
        - **Random Forest**: Ensemble method for better performance
        - **Support Vector Machine**: Effective for high-dimensional data
        
        ### Features
        
        ✅ **Text Preprocessing**: Advanced text cleaning and normalization  
        ✅ **Feature Extraction**: TF-IDF vectorization with additional features  
        ✅ **Multiple Models**: Compare different ML algorithms  
        ✅ **Real-time Prediction**: Instant spam detection  
        ✅ **Interactive Interface**: User-friendly web application  
        ✅ **Model Evaluation**: Comprehensive performance metrics  
        
        ### How to Use
        
        1. **Train Models**: Go to "Train Models" to train the spam detection models
        2. **Make Predictions**: Use "Predict" to classify new messages
        3. **View Analysis**: Check "Analysis" for dataset insights
        4. **Review Results**: See "Results" for model performance
        """)
    
    with col2:
        st.markdown("### Quick Stats")
        
        # Check if models exist
        if os.path.exists('models'):
            st.success("✅ Models Trained")
            predictor = SpamPredictor()
            if predictor.models:
                summary = predictor.get_model_performance_summary()
                st.metric("Available Models", summary['total_models'])
                st.metric("Best Model", summary['best_model'])
        else:
            st.warning("⚠️ Models Not Trained")
            st.info("Go to 'Train Models' to get started")
        
        # Check if dataset exists
        try:
            df = get_dataset()
            if df is not None:
                st.metric("Total Messages", len(df))
                st.metric("Spam Messages", len(df[df['label'] == 'spam']))
                st.metric("Ham Messages", len(df[df['label'] == 'ham']))
        except:
            st.info("Dataset not loaded")

def show_predict_page():
    """
    Display the prediction page
    """
    st.markdown("## 🎯 SMS Spam Prediction")
    
    # Check if models are available
    if not os.path.exists('models'):
        st.error("❌ No trained models found!")
        st.info("Please train models first by going to 'Train Models' page.")
        return
    
    predictor = SpamPredictor()
    if not predictor.models:
        st.error("❌ Failed to load models!")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Prediction interface
    st.markdown("### Enter SMS Message")
    
    # Text input
    message = st.text_area(
        "Type or paste your SMS message here:",
        height=100,
        placeholder="Enter your message here..."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Choose Model:",
            ["Best Model"] + list(predictor.models.keys()),
            help="Select which model to use for prediction"
        )
    
    with col2:
        if st.button("🔍 Predict", type="primary"):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    try:
                        # Make prediction
                        if model_choice == "Best Model":
                            result = predictor.predict_single(message)
                        else:
                            result = predictor.predict_single(message, model_choice)
                        
                        explanation = predictor.explain_prediction(message, model_choice if model_choice != "Best Model" else None)
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        # Prediction card
                        prediction_class = "spam-prediction" if result['prediction'] == 'spam' else "ham-prediction"
                        prediction_color = "#ff4444" if result['prediction'] == 'spam' else "#44ff44"
                        prediction_emoji = "🚨" if result['prediction'] == 'spam' else "✅"
                        
                        st.markdown(f"""
                        <div class="prediction-card {prediction_class}">
                            <h3>{prediction_emoji} {result['prediction'].upper()}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.3f}</p>
                            <p><strong>Model Used:</strong> {result['model_used']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['confidence'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Confidence Level (%)"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': prediction_color},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explanation
                        st.markdown("### 🔍 Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Key Features:**")
                            features = explanation['key_features']
                            for key, value in features.items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                        
                        with col2:
                            st.markdown("**Spam Indicators:**")
                            if explanation['spam_indicators']:
                                for indicator in explanation['spam_indicators']:
                                    st.write(f"- {indicator}")
                            else:
                                st.write("- No obvious spam indicators detected")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")
            else:
                st.warning("⚠️ Please enter a message to predict.")
    
    # Batch prediction
    st.markdown("---")
    st.markdown("### 📋 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with messages (should have a 'message' column)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'message' in df_upload.columns:
                st.success(f"✅ Loaded {len(df_upload)} messages")
                
                if st.button("🔍 Predict All"):
                    with st.spinner("Processing messages..."):
                        results = predictor.predict_batch(df_upload['message'].tolist())
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        results_df = results_df[['message', 'prediction', 'confidence', 'model_used']]
                        
                        st.markdown("### 📊 Batch Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="spam_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Messages", len(results_df))
                        with col2:
                            st.metric("Spam Detected", len(results_df[results_df['prediction'] == 'spam']))
                        with col3:
                            st.metric("Ham Detected", len(results_df[results_df['prediction'] == 'ham']))
            else:
                st.error("❌ CSV file must contain a 'message' column")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def show_analysis_page():
    """
    Display the data analysis page
    """
    st.markdown("## 📊 Data Analysis")
    
    try:
        df = get_dataset()
        if df is None:
            st.error("❌ Failed to load dataset!")
            return
        
        st.success(f"✅ Dataset loaded: {len(df)} messages")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Spam Messages", len(df[df['label'] == 'spam']))
        with col3:
            st.metric("Ham Messages", len(df[df['label'] == 'ham']))
        with col4:
            spam_ratio = len(df[df['label'] == 'spam']) / len(df) * 100
            st.metric("Spam Ratio", f"{spam_ratio:.1f}%")
        
        # Label distribution
        st.markdown("### 📈 Label Distribution")
        fig = px.pie(
            df, 
            names='label', 
            title="Distribution of Spam vs Ham Messages",
            color_discrete_map={'spam': '#ff4444', 'ham': '#44ff44'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Text length analysis
        st.markdown("### 📏 Message Length Analysis")
        
        # Preprocess data for analysis
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text length distribution
            fig = px.histogram(
                df_processed,
                x='text_length',
                color='label',
                title="Message Length Distribution",
                color_discrete_map={'spam': '#ff4444', 'ham': '#44ff44'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Word count distribution
            fig = px.histogram(
                df_processed,
                x='word_count',
                color='label',
                title="Word Count Distribution",
                color_discrete_map={'spam': '#ff4444', 'ham': '#44ff44'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.markdown("### 🔍 Feature Analysis")
        
        features = ['uppercase_count', 'digit_count', 'special_char_count', 'spam_indicators']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f.replace('_', ' ').title() for f in features]
        )
        
        for i, feature in enumerate(features):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Box(
                    y=df_processed[df_processed['label'] == 'spam'][feature],
                    name='Spam',
                    marker_color='#ff4444'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Box(
                    y=df_processed[df_processed['label'] == 'ham'][feature],
                    name='Ham',
                    marker_color='#44ff44'
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Feature Distributions by Label")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample messages
        st.markdown("### 📝 Sample Messages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Spam Messages:**")
            spam_samples = df[df['label'] == 'spam']['message'].head(5).tolist()
            for i, msg in enumerate(spam_samples, 1):
                st.text_area(f"Spam {i}", msg, height=80, key=f"spam_{i}")
        
        with col2:
            st.markdown("**Sample Ham Messages:**")
            ham_samples = df[df['label'] == 'ham']['message'].head(5).tolist()
            for i, msg in enumerate(ham_samples, 1):
                st.text_area(f"Ham {i}", msg, height=80, key=f"ham_{i}")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")

def show_train_page():
    """
    Display the model training page
    """
    st.markdown("## 🤖 Train Models")
    
    st.markdown("""
    This page allows you to train the SMS spam detection models. 
    The training process includes:
    
    1. **Data Loading**: Load the SMS dataset
    2. **Preprocessing**: Clean and prepare the text data
    3. **Feature Extraction**: Convert text to numerical features
    4. **Model Training**: Train multiple ML algorithms
    5. **Evaluation**: Compare model performance
    """)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Import and run training
                from main import run_training_pipeline
                
                # Capture output
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    run_training_pipeline()
                
                output = f.getvalue()
                
                # Display results
                st.success("✅ Training completed successfully!")
                st.text(output)
                
                # Show model files
                if os.path.exists('models'):
                    st.markdown("### 📁 Generated Model Files")
                    model_files = os.listdir('models')
                    for file in model_files:
                        st.write(f"- {file}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    
    # Training options
    st.markdown("---")
    st.markdown("### ⚙️ Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Use Cross-Validation", value=True)
        st.checkbox("Save Model Plots", value=True)
    
    with col2:
        st.selectbox("Test Size", [0.1, 0.2, 0.3], index=1)
        st.selectbox("Random State", [42, 123, 456], index=0)

def show_results_page():
    """
    Display the results page
    """
    st.markdown("## 📈 Model Results")
    
    if not os.path.exists('models'):
        st.error("❌ No trained models found!")
        st.info("Please train models first by going to 'Train Models' page.")
        return
    
    try:
        # Load models and results
        predictor = SpamPredictor()
        
        if not predictor.models:
            st.error("❌ Failed to load models!")
            return
        
        st.success("✅ Models loaded successfully!")
        
        # Model summary
        st.markdown("### 📊 Model Summary")
        summary = predictor.get_model_performance_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", summary['total_models'])
        with col2:
            st.metric("Best Model", summary['best_model'])
        with col3:
            st.metric("Feature Extractor", "✅ Loaded" if summary['feature_extractor_loaded'] else "❌ Missing")
        
        # Model comparison (if results exist)
        if os.path.exists('models/model_comparison.png'):
            st.markdown("### 📊 Model Performance Comparison")
            st.image('models/model_comparison.png', use_column_width=True)
        
        if os.path.exists('models/confusion_matrices.png'):
            st.markdown("### 🎯 Confusion Matrices")
            st.image('models/confusion_matrices.png', use_column_width=True)
        
        # Model files
        st.markdown("### 📁 Model Files")
        model_files = os.listdir('models')
        for file in model_files:
            file_path = os.path.join('models', file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            st.write(f"- {file} ({file_size:.1f} KB)")
        
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    main() 