"""
Data Collection Module for SMS Spam Detection
Downloads and prepares the SMS Spam Collection Dataset
"""

import pandas as pd
import requests
import os
from pathlib import Path

def download_sms_dataset():
    """
    Download the SMS Spam Collection Dataset from UCI repository
    """
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset
    print("Downloading SMS Spam Collection Dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the zip file
        zip_path = data_dir / "smsspamcollection.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Dataset downloaded successfully to {zip_path}")
        return zip_path
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_sms_data():
    """
    Load the SMS dataset from the data directory
    If not present, download it first
    """
    data_dir = Path("data")
    sms_file = data_dir / "SMSSpamCollection"
    
    # If dataset doesn't exist, download it
    if not sms_file.exists():
        print("Dataset not found. Downloading...")
        zip_path = download_sms_dataset()
        if zip_path and zip_path.exists():
            # Extract the zip file (you might need to do this manually or use zipfile module)
            print("Please extract the downloaded zip file manually to the data directory")
            print(f"Extract {zip_path} to {data_dir}")
            return None
    
    # Load the dataset
    try:
        # The dataset has tab-separated values with no header
        df = pd.read_csv(sms_file, sep='\t', names=['label', 'message'])
        print(f"Dataset loaded successfully: {len(df)} messages")
        print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
        print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
        return df
        
    except FileNotFoundError:
        print("SMS dataset file not found. Please ensure the file is extracted correctly.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_sample_dataset():
    """
    Create a sample dataset for testing if the main dataset is not available
    """
    sample_data = {
        'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham'],
        'message': [
            "Hey there! How are you doing?",
            "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net",
            "Ok lar... Joking wif u oni...",
            "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply",
            "Even my brother is not like to speak with me. They treat me like aids patent.",
            "URGENT! Your Mobile No. was awarded £200 Bonus Caller Prize on 1/08. This is our 2nd attempt to contact YOU! Call 09066362231 ASAP! Box97N7QP, 150ppm",
            "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
            "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample dataset created for testing")
    print(f"Sample dataset: {len(df)} messages")
    return df

def get_dataset():
    """
    Main function to get the SMS dataset
    Returns the dataset as a pandas DataFrame
    """
    # Try to load the real dataset first
    df = load_sms_data()
    
    if df is not None:
        return df
    else:
        print("Using sample dataset for demonstration...")
        return create_sample_dataset()

if __name__ == "__main__":
    # Test the data collection
    df = get_dataset()
    if df is not None:
        print("\nDataset Preview:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}") 