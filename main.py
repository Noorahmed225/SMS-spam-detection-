"""
Main Execution Script for SMS Spam Detection
Orchestrates the complete pipeline from data loading to model training and evaluation
"""

import sys
import os
from pathlib import Path
import argparse
import time

# Add src directory to path
sys.path.append('src')

from data_collection import get_dataset
from preprocessing import TextPreprocessor
from feature_extraction import prepare_features
from model_training import train_and_evaluate
from prediction import SpamPredictor, create_sample_predictions

def main():
    """
    Main function to run the complete SMS spam detection pipeline
    """
    parser = argparse.ArgumentParser(description='SMS Spam Detection Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'demo'], 
                       default='train', help='Mode to run the pipeline')
    parser.add_argument('--message', type=str, help='Message to predict (for predict mode)')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode for predictions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMS SPAM DETECTION PROJECT")
    print("=" * 60)
    
    if args.mode == 'train':
        run_training_pipeline()
    elif args.mode == 'predict':
        run_prediction_pipeline(args.message, args.interactive)
    elif args.mode == 'demo':
        run_demo()
    else:
        print("Invalid mode. Use --help for options.")

def run_training_pipeline():
    """
    Run the complete training pipeline
    """
    print("\n🚀 Starting SMS Spam Detection Training Pipeline")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Data Collection
    print("\n📊 Step 1: Loading Dataset")
    print("-" * 30)
    df = get_dataset()
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Total messages: {len(df)}")
    print(f"   - Spam messages: {len(df[df['label'] == 'spam'])}")
    print(f"   - Ham messages: {len(df[df['label'] == 'ham'])}")
    
    # Step 2: Data Preprocessing
    print("\n🔧 Step 2: Data Preprocessing")
    print("-" * 30)
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    print("✅ Data preprocessing completed!")
    print(f"   - Added processed text column")
    print(f"   - Extracted {len([col for col in df_processed.columns if col not in ['label', 'message', 'processed_text']])} additional features")
    
    # Step 3: Feature Extraction
    print("\n⚙️  Step 3: Feature Extraction")
    print("-" * 30)
    numerical_columns = ['text_length', 'word_count', 'char_count', 'avg_word_length', 
                        'uppercase_count', 'digit_count', 'special_char_count', 'spam_indicators']
    
    X_train, X_test, y_train, y_test, feature_extractor = prepare_features(
        df_processed, numerical_columns=numerical_columns
    )
    print("✅ Feature extraction completed!")
    print(f"   - Training set: {X_train.shape}")
    print(f"   - Testing set: {X_test.shape}")
    print(f"   - Total features: {X_train.shape[1]}")
    
    # Step 4: Model Training and Evaluation
    print("\n🤖 Step 4: Model Training and Evaluation")
    print("-" * 30)
    classifier = train_and_evaluate(X_train, X_test, y_train, y_test, feature_extractor)
    print("✅ Model training completed!")
    print(f"   - Best model: {classifier.best_model_name}")
    print(f"   - Models saved to 'models/' directory")
    
    # Step 5: Generate Sample Predictions
    print("\n🎯 Step 5: Testing Predictions")
    print("-" * 30)
    create_sample_predictions()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("   1. Run 'python main.py --mode predict --interactive' for interactive predictions")
    print("   2. Run 'python main.py --mode predict --message \"Your message here\"' for single prediction")
    print("   3. Run 'streamlit run app.py' for web interface")

def run_prediction_pipeline(message=None, interactive=False):
    """
    Run the prediction pipeline
    """
    print("\n🎯 SMS Spam Detection - Prediction Mode")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SpamPredictor()
    
    if not predictor.models:
        print("❌ No trained models found!")
        print("   Please run training first: python main.py --mode train")
        return
    
    print("✅ Models loaded successfully!")
    summary = predictor.get_model_performance_summary()
    print(f"   - Available models: {summary['available_models']}")
    print(f"   - Best model: {summary['best_model']}")
    
    if interactive:
        print("\n🔍 Interactive Prediction Mode")
        print("Type 'quit' to exit")
        print("-" * 30)
        
        while True:
            user_message = input("\nEnter SMS message: ").strip()
            
            if user_message.lower() == 'quit':
                break
            
            if not user_message:
                print("Please enter a message.")
                continue
            
            try:
                result = predictor.predict_single(user_message)
                explanation = predictor.explain_prediction(user_message)
                
                print(f"\n📱 Message: {user_message}")
                print(f"🎯 Prediction: {result['prediction'].upper()}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"🤖 Model: {result['model_used']}")
                
                if explanation['spam_indicators']:
                    print(f"⚠️  Spam indicators: {', '.join(explanation['spam_indicators'])}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif message:
        print(f"\n📱 Predicting for message: {message}")
        print("-" * 30)
        
        try:
            result = predictor.predict_single(message)
            explanation = predictor.explain_prediction(message)
            
            print(f"🎯 Prediction: {result['prediction'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"🤖 Model: {result['model_used']}")
            
            if explanation['spam_indicators']:
                print(f"⚠️  Spam indicators: {', '.join(explanation['spam_indicators'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print("\n📋 Sample Predictions:")
        print("-" * 30)
        create_sample_predictions()

def run_demo():
    """
    Run a demonstration of the complete pipeline
    """
    print("\n🎬 SMS Spam Detection - Demo Mode")
    print("=" * 50)
    
    # Check if models exist
    if not os.path.exists('models'):
        print("❌ No trained models found!")
        print("   Running training pipeline first...")
        run_training_pipeline()
    else:
        print("✅ Trained models found!")
    
    print("\n🎯 Running Sample Predictions:")
    print("-" * 30)
    create_sample_predictions()
    
    print("\n🔍 Interactive Demo:")
    print("   Run 'python main.py --mode predict --interactive' for interactive predictions")
    print("   Run 'streamlit run app.py' for web interface")

if __name__ == "__main__":
    main() 