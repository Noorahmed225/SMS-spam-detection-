"""
Setup Script for SMS Spam Detection Project
Helps users install dependencies and set up the project
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = ['data', 'models', 'notebooks']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_setup():
    """Test if setup was successful"""
    print("Testing setup...")
    try:
        # Test imports
        sys.path.append('src')
        from data_collection import get_dataset
        from preprocessing import TextPreprocessor
        from feature_extraction import FeatureExtractor
        from model_training import SpamClassifier
        from prediction import SpamPredictor
        
        print("✅ All modules imported successfully!")
        
        # Test dataset loading
        df = get_dataset()
        if df is not None:
            print(f"✅ Dataset loaded: {len(df)} messages")
        else:
            print("⚠️  Dataset not loaded (will use sample data)")
        
        return True
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("SMS Spam Detection Project - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Test setup
    if not test_setup():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python test_project.py' to verify everything works")
    print("2. Run 'python main.py --mode train' to train models")
    print("3. Run 'python main.py --mode predict --interactive' for predictions")
    print("4. Run 'streamlit run app.py' for web interface")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 