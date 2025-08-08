"""
Text Preprocessing Module for SMS Spam Detection
Handles text cleaning, tokenization, and normalization
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A class for preprocessing SMS text data
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (various formats)
        text = re.sub(r'[\+]?[1-9][\d]{0,15}', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """
        Lemmatize words in text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def extract_features(self, text):
        """
        Extract additional features from text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Text length
        features['text_length'] = len(text)
        
        # Word count
        features['word_count'] = len(text.split())
        
        # Character count (excluding spaces)
        features['char_count'] = len(text.replace(' ', ''))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0
        
        # Count of uppercase letters
        features['uppercase_count'] = sum(1 for char in text if char.isupper())
        
        # Count of digits
        features['digit_count'] = sum(1 for char in text if char.isdigit())
        
        # Count of special characters
        features['special_char_count'] = sum(1 for char in text if char in string.punctuation)
        
        # Check for common spam indicators
        spam_indicators = ['free', 'urgent', 'winner', 'prize', 'cash', 'claim', 'call now', 
                          'limited time', 'offer', 'discount', 'money', 'guaranteed']
        features['spam_indicators'] = sum(1 for indicator in spam_indicators if indicator in text.lower())
        
        return features
    
    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Complete text preprocessing pipeline
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize words
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            cleaned_text = self.remove_stopwords(cleaned_text)
        
        # Lemmatize if requested
        if lemmatize:
            cleaned_text = self.lemmatize_text(cleaned_text)
        
        return cleaned_text
    
    def preprocess_dataframe(self, df, text_column='message', remove_stopwords=True, lemmatize=True):
        """
        Preprocess all text in a DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize words
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        df_processed = df.copy()
        
        # Add preprocessed text column
        df_processed['processed_text'] = df_processed[text_column].apply(
            lambda x: self.preprocess_text(x, remove_stopwords, lemmatize)
        )
        
        # Extract additional features
        feature_dfs = []
        for idx, row in df_processed.iterrows():
            features = self.extract_features(row[text_column])
            feature_dfs.append(pd.Series(features))
        
        features_df = pd.DataFrame(feature_dfs)
        df_processed = pd.concat([df_processed, features_df], axis=1)
        
        return df_processed

def create_vocabulary(texts, min_freq=2):
    """
    Create vocabulary from a list of texts
    
    Args:
        texts (list): List of text strings
        min_freq (int): Minimum frequency for a word to be included in vocabulary
        
    Returns:
        dict: Vocabulary with word frequencies
    """
    word_freq = {}
    
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by minimum frequency
    vocabulary = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    return vocabulary

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = TextPreprocessor()
    
    # Test text
    test_text = "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word: CLAIM to No: 81010"
    
    print("Original text:", test_text)
    print("Cleaned text:", preprocessor.clean_text(test_text))
    print("Without stopwords:", preprocessor.remove_stopwords(preprocessor.clean_text(test_text)))
    print("Lemmatized:", preprocessor.lemmatize_text(preprocessor.remove_stopwords(preprocessor.clean_text(test_text))))
    print("Complete preprocessing:", preprocessor.preprocess_text(test_text))
    print("Extracted features:", preprocessor.extract_features(test_text)) 