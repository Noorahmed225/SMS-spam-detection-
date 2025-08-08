"""
Feature Extraction Module for SMS Spam Detection
Converts text data into numerical features for machine learning
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class FeatureExtractor:
    """
    A class for extracting features from SMS text data
    """
    
    def __init__(self, vectorizer_type='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the feature extractor
        
        Args:
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        
        # Initialize scaler for numerical features (use MinMaxScaler to avoid negative values)
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Store feature names
        self.text_feature_names = None
        self.numerical_feature_names = None
        
    def fit_transform(self, texts, numerical_features=None, labels=None):
        """
        Fit the vectorizer and transform the data
        
        Args:
            texts (list): List of text strings
            numerical_features (pd.DataFrame): Numerical features
            labels (list): Target labels
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
        """
        # Fit and transform text features
        print("Extracting text features...")
        text_features = self.vectorizer.fit_transform(texts)
        self.text_feature_names = self.vectorizer.get_feature_names_out()
        
        # Combine with numerical features if provided
        if numerical_features is not None:
            print("Scaling numerical features...")
            scaled_numerical = self.scaler.fit_transform(numerical_features)
            self.numerical_feature_names = numerical_features.columns.tolist()
            
            # Convert sparse matrix to dense if needed
            if hasattr(text_features, 'toarray'):
                text_features_dense = text_features.toarray()
            else:
                text_features_dense = text_features
            
            # Combine features
            X = np.hstack([text_features_dense, scaled_numerical])
        else:
            X = text_features
        
        # Encode labels if provided
        if labels is not None:
            y = self.label_encoder.fit_transform(labels)
            return X, y
        else:
            return X
    
    def transform(self, texts, numerical_features=None):
        """
        Transform new data using fitted vectorizer
        
        Args:
            texts (list): List of text strings
            numerical_features (pd.DataFrame): Numerical features
            
        Returns:
            np.ndarray: Feature matrix
        """
        # Transform text features
        text_features = self.vectorizer.transform(texts)
        
        # Combine with numerical features if provided
        if numerical_features is not None:
            scaled_numerical = self.scaler.transform(numerical_features)
            
            # Convert sparse matrix to dense if needed
            if hasattr(text_features, 'toarray'):
                text_features_dense = text_features.toarray()
            else:
                text_features_dense = text_features
            
            # Combine features
            X = np.hstack([text_features_dense, scaled_numerical])
        else:
            X = text_features
        
        return X
    
    def get_feature_names(self):
        """
        Get feature names
        
        Returns:
            list: List of feature names
        """
        feature_names = []
        
        if self.text_feature_names is not None:
            feature_names.extend(self.text_feature_names)
        
        if self.numerical_feature_names is not None:
            feature_names.extend(self.numerical_feature_names)
        
        return feature_names
    
    def save(self, filepath):
        """
        Save the fitted feature extractor
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'text_feature_names': self.text_feature_names,
            'numerical_feature_names': self.numerical_feature_names,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Feature extractor saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a fitted feature extractor
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.text_feature_names = model_data['text_feature_names']
        self.numerical_feature_names = model_data['numerical_feature_names']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        
        print(f"Feature extractor loaded from {filepath}")

def prepare_features(df, text_column='processed_text', label_column='label', 
                    numerical_columns=None, test_size=0.2, random_state=42):
    """
    Prepare features for training
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        numerical_columns (list): List of numerical column names
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_extractor)
    """
    # Extract texts and labels
    texts = df[text_column].fillna('').tolist()
    labels = df[label_column].tolist()
    
    # Extract numerical features if specified
    numerical_features = None
    if numerical_columns:
        numerical_features = df[numerical_columns].fillna(0)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Fit and transform the data
    X, y = feature_extractor.fit_transform(texts, numerical_features, labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_extractor

def extract_text_features_only(df, text_column='processed_text', label_column='label',
                              test_size=0.2, random_state=42, max_features=5000):
    """
    Extract only text features (no numerical features)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_extractor)
    """
    return prepare_features(
        df, text_column, label_column, 
        numerical_columns=None, 
        test_size=test_size, 
        random_state=random_state
    )

if __name__ == "__main__":
    # Test the feature extraction
    from data_collection import get_dataset
    from preprocessing import TextPreprocessor
    
    # Load and preprocess data
    df = get_dataset()
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Extract features
    X_train, X_test, y_train, y_test, feature_extractor = prepare_features(
        df_processed, 
        numerical_columns=['text_length', 'word_count', 'char_count', 'avg_word_length', 
                          'uppercase_count', 'digit_count', 'special_char_count', 'spam_indicators']
    )
    
    print("Feature extraction completed successfully!")
    print(f"Feature names: {len(feature_extractor.get_feature_names())}") 