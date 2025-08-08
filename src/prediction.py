"""
Prediction Module for SMS Spam Detection
Makes predictions on new SMS messages using trained models
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor

class SpamPredictor:
    """
    A class for making predictions on new SMS messages
    """
    
    def __init__(self, model_path='models'):
        """
        Initialize the predictor with trained models
        
        Args:
            model_path: Path to the directory containing trained models
        """
        self.model_path = model_path
        self.models = {}
        self.feature_extractor = None
        self.preprocessor = TextPreprocessor()
        self.best_model = None
        self.best_model_name = None
        
        # Load models if they exist
        self.load_models()
    
    def load_models(self):
        """
        Load trained models from the specified directory
        """
        if not os.path.exists(self.model_path):
            print(f"Model directory {self.model_path} not found.")
            return
        
        # Load feature extractor
        fe_path = os.path.join(self.model_path, 'feature_extractor.pkl')
        if os.path.exists(fe_path):
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.load(fe_path)
            print("Feature extractor loaded successfully")
        
        # Load best model info
        best_model_path = os.path.join(self.model_path, 'best_model_info.pkl')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                best_model_info = pickle.load(f)
            self.best_model_name = best_model_info['best_model_name']
            self.best_model = best_model_info['best_model']
            print(f"Best model loaded: {self.best_model_name}")
        
        # Load individual models
        model_names = ['naive_bayes', 'logistic_regression', 'random_forest', 'svm']
        for name in model_names:
            model_file = os.path.join(self.model_path, f'{name}_model.pkl')
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Model {name} loaded successfully")
    
    def preprocess_message(self, message):
        """
        Preprocess a single message
        
        Args:
            message (str): Input message
            
        Returns:
            dict: Dictionary containing preprocessed text and extracted features
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(message)
        
        # Extract features
        features = self.preprocessor.extract_features(message)
        
        return {
            'processed_text': processed_text,
            'features': features
        }
    
    def predict_single(self, message, model_name=None):
        """
        Predict spam/ham for a single message
        
        Args:
            message (str): Input message
            model_name (str): Name of the model to use (if None, uses best model)
            
        Returns:
            dict: Prediction results
        """
        if not self.feature_extractor:
            raise ValueError("Feature extractor not loaded. Please train models first.")
        
        # Preprocess message
        preprocessed = self.preprocess_message(message)
        
        # Prepare features
        texts = [preprocessed['processed_text']]
        numerical_features = pd.DataFrame([preprocessed['features']])
        
        # Transform features
        X = self.feature_extractor.transform(texts, numerical_features)
        
        # Make prediction
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.best_model:
            model = self.best_model
            model_name = self.best_model_name
        else:
            raise ValueError("No model available for prediction")
        
        # Get prediction and probability
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        if self.feature_extractor.label_encoder:
            prediction_label = self.feature_extractor.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = 'spam' if prediction == 1 else 'ham'
        
        # Calculate confidence
        if probability is not None:
            confidence = max(probability)
        else:
            confidence = None
        
        return {
            'message': message,
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': probability,
            'model_used': model_name,
            'preprocessed_text': preprocessed['processed_text'],
            'extracted_features': preprocessed['features']
        }
    
    def predict_batch(self, messages, model_name=None):
        """
        Predict spam/ham for multiple messages
        
        Args:
            messages (list): List of input messages
            model_name (str): Name of the model to use (if None, uses best model)
            
        Returns:
            list: List of prediction results
        """
        if not self.feature_extractor:
            raise ValueError("Feature extractor not loaded. Please train models first.")
        
        results = []
        
        for message in messages:
            result = self.predict_single(message, model_name)
            results.append(result)
        
        return results
    
    def predict_with_all_models(self, message):
        """
        Predict using all available models
        
        Args:
            message (str): Input message
            
        Returns:
            dict: Predictions from all models
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                result = self.predict_single(message, name)
                results[name] = {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def get_model_performance_summary(self):
        """
        Get a summary of available models
        
        Returns:
            dict: Summary of available models
        """
        summary = {
            'total_models': len(self.models),
            'best_model': self.best_model_name,
            'available_models': list(self.models.keys()),
            'feature_extractor_loaded': self.feature_extractor is not None
        }
        
        return summary
    
    def explain_prediction(self, message, model_name=None):
        """
        Provide explanation for a prediction
        
        Args:
            message (str): Input message
            model_name (str): Name of the model to use
            
        Returns:
            dict: Explanation of the prediction
        """
        # Get prediction
        prediction_result = self.predict_single(message, model_name)
        
        # Extract features
        features = prediction_result['extracted_features']
        
        # Identify key indicators
        spam_indicators = []
        if features['spam_indicators'] > 0:
            spam_indicators.append(f"Contains {features['spam_indicators']} spam indicator words")
        
        if features['uppercase_count'] > len(message) * 0.3:
            spam_indicators.append("High percentage of uppercase letters")
        
        if features['digit_count'] > 5:
            spam_indicators.append("Contains many digits")
        
        if features['special_char_count'] > 3:
            spam_indicators.append("Contains special characters")
        
        if features['text_length'] > 100:
            spam_indicators.append("Long message")
        
        # Create explanation
        explanation = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'key_features': {
                'text_length': features['text_length'],
                'word_count': features['word_count'],
                'uppercase_count': features['uppercase_count'],
                'digit_count': features['digit_count'],
                'spam_indicators': features['spam_indicators']
            },
            'spam_indicators': spam_indicators,
            'model_used': prediction_result['model_used']
        }
        
        return explanation

def create_sample_predictions():
    """
    Create sample predictions for demonstration
    """
    # Sample messages
    sample_messages = [
        "Hey there! How are you doing?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word: CLAIM to No: 81010",
        "Ok lar... Joking wif u oni...",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "URGENT! Your Mobile No. was awarded £200 Bonus Caller Prize on 1/08. This is our 2nd attempt to contact YOU! Call 09066362231 ASAP!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
        "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"
    ]
    
    # Initialize predictor
    predictor = SpamPredictor()
    
    # Check if models are loaded
    if not predictor.models:
        print("No trained models found. Please train models first.")
        return
    
    print("Sample Predictions:")
    print("=" * 60)
    
    for i, message in enumerate(sample_messages, 1):
        print(f"\n{i}. Message: {message[:50]}...")
        
        try:
            result = predictor.predict_single(message)
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Model: {result['model_used']}")
            
            # Get explanation
            explanation = predictor.explain_prediction(message)
            if explanation['spam_indicators']:
                print(f"   Indicators: {', '.join(explanation['spam_indicators'])}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)

def interactive_prediction():
    """
    Interactive prediction interface
    """
    predictor = SpamPredictor()
    
    if not predictor.models:
        print("No trained models found. Please train models first.")
        return
    
    print("SMS Spam Detection - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        message = input("\nEnter SMS message: ").strip()
        
        if message.lower() == 'quit':
            break
        
        if not message:
            print("Please enter a message.")
            continue
        
        try:
            result = predictor.predict_single(message)
            explanation = predictor.explain_prediction(message)
            
            print(f"\nPrediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Model: {result['model_used']}")
            
            if explanation['spam_indicators']:
                print(f"Spam indicators: {', '.join(explanation['spam_indicators'])}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Test the prediction functionality
    print("Testing SMS Spam Prediction...")
    
    # Create sample predictions
    create_sample_predictions()
    
    # Uncomment the line below to run interactive mode
    # interactive_prediction() 