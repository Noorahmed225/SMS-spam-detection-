"""
SMS Spam Detection - Exploratory Data Analysis
This script provides a comprehensive analysis of the SMS Spam Collection Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import sys
import os

# Add src to path
sys.path.append('../src')

from data_collection import get_dataset
from preprocessing import TextPreprocessor

def main():
    """Main analysis function"""
    print("SMS Spam Detection - Exploratory Data Analysis")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    df = get_dataset()
    
    if df is None:
        print("Failed to load dataset")
        return
    
    print(f"Dataset loaded: {len(df)} messages")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"- Total messages: {len(df)}")
    print(f"- Spam messages: {len(df[df['label'] == 'spam'])}")
    print(f"- Ham messages: {len(df[df['label'] == 'ham'])}")
    print(f"- Spam ratio: {len(df[df['label'] == 'spam']) / len(df) * 100:.2f}%")
    
    # Text analysis
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    
    print("\nMessage Length Statistics:")
    print(df.groupby('label')['message_length'].describe())
    
    # Feature extraction
    print("\nExtracting features...")
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print("Features extracted successfully!")
    print(f"Additional features: {[col for col in df_processed.columns if col not in ['label', 'message', 'processed_text']]}")
    
    # Save processed data
    df_processed.to_csv('../data/processed_sms_data.csv', index=False)
    print("\nProcessed data saved to ../data/processed_sms_data.csv")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 