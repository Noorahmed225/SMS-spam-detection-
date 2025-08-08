"""
Model Training Module for SMS Spam Detection
Trains and evaluates multiple machine learning models
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

class SpamClassifier:
    """
    A class for training and evaluating spam detection models
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_extractor = None
        
    def initialize_models(self):
        """
        Initialize different machine learning models
        """
        self.models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        print("Models initialized:")
        for name, model in self.models.items():
            print(f"  - {name}: {type(model).__name__}")
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_extractor=None):
        """
        Train all models and evaluate their performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            feature_extractor: Feature extractor object
        """
        self.feature_extractor = feature_extractor
        results = {}
        
        print("Training models...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate ROC AUC if possible
            roc_auc = None
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Find the best model
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"Best F1-Score: {results[self.best_model_name]['f1_score']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model to tune
            
        Returns:
            Best model after tuning
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def cross_validation(self, X, y, cv=5):
        """
        Perform cross-validation for all models
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        cv_results = {}
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        for name, model in self.models.items():
            print(f"\nCross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"  Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def plot_results(self, results, save_path=None):
        """
        Plot model comparison results
        
        Args:
            results: Dictionary with model results
            save_path: Path to save the plot
        """
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            values = [results[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, results, y_test, save_path=None):
        """
        Plot confusion matrices for all models
        
        Args:
            results: Dictionary with model results
            y_test: True test labels
            save_path: Path to save the plot
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confusion Matrices', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            cm = confusion_matrix(y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def save_models(self, directory='models'):
        """
        Save all trained models
        
        Args:
            directory: Directory to save models
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(directory, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model {name} saved to {model_path}")
        
        # Save feature extractor if available
        if self.feature_extractor:
            fe_path = os.path.join(directory, 'feature_extractor.pkl')
            self.feature_extractor.save(fe_path)
        
        # Save best model info
        best_model_info = {
            'best_model_name': self.best_model_name,
            'best_model': self.best_model
        }
        best_model_path = os.path.join(directory, 'best_model_info.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model_info, f)
        
        print(f"Best model info saved to {best_model_path}")
    
    def load_models(self, directory='models'):
        """
        Load trained models
        
        Args:
            directory: Directory containing saved models
        """
        # Load individual models
        for name in self.models.keys():
            model_path = os.path.join(directory, f'{name}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Model {name} loaded from {model_path}")
        
        # Load feature extractor if available
        fe_path = os.path.join(directory, 'feature_extractor.pkl')
        if os.path.exists(fe_path):
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.load(fe_path)
        
        # Load best model info
        best_model_path = os.path.join(directory, 'best_model_info.pkl')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'rb') as f:
                best_model_info = pickle.load(f)
            self.best_model_name = best_model_info['best_model_name']
            self.best_model = best_model_info['best_model']
            print(f"Best model info loaded: {self.best_model_name}")

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_extractor=None):
    """
    Complete training and evaluation pipeline
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        feature_extractor: Feature extractor object
        
    Returns:
        Trained classifier object
    """
    # Initialize classifier
    classifier = SpamClassifier()
    classifier.initialize_models()
    
    # Train models
    results = classifier.train_models(X_train, y_train, X_test, y_test, feature_extractor)
    
    # Perform cross-validation
    cv_results = classifier.cross_validation(X_train, y_train)
    
    # Plot results
    classifier.plot_results(results, 'models/model_comparison.png')
    classifier.plot_confusion_matrices(results, y_test, 'models/confusion_matrices.png')
    
    # Save models
    classifier.save_models()
    
    return classifier

if __name__ == "__main__":
    # Test the model training
    from data_collection import get_dataset
    from preprocessing import TextPreprocessor
    from feature_extraction import prepare_features
    
    # Load and preprocess data
    print("Loading dataset...")
    df = get_dataset()
    
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print("Extracting features...")
    X_train, X_test, y_train, y_test, feature_extractor = prepare_features(
        df_processed,
        numerical_columns=['text_length', 'word_count', 'char_count', 'avg_word_length', 
                          'uppercase_count', 'digit_count', 'special_char_count', 'spam_indicators']
    )
    
    # Train and evaluate models
    print("Training models...")
    classifier = train_and_evaluate(X_train, X_test, y_train, y_test, feature_extractor)
    
    print("Training completed successfully!") 