"""
Test Script for SMS Spam Detection Project
Verifies that all components work correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_collection import get_dataset
        print("✅ data_collection imported successfully")
    except Exception as e:
        print(f"❌ data_collection import failed: {e}")
        return False
    
    try:
        from preprocessing import TextPreprocessor
        print("✅ preprocessing imported successfully")
    except Exception as e:
        print(f"❌ preprocessing import failed: {e}")
        return False
    
    try:
        from feature_extraction import FeatureExtractor
        print("✅ feature_extraction imported successfully")
    except Exception as e:
        print(f"❌ feature_extraction import failed: {e}")
        return False
    
    try:
        from model_training import SpamClassifier
        print("✅ model_training imported successfully")
    except Exception as e:
        print(f"❌ model_training import failed: {e}")
        return False
    
    try:
        from prediction import SpamPredictor
        print("✅ prediction imported successfully")
    except Exception as e:
        print(f"❌ prediction import failed: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality"""
    print("\nTesting data collection...")
    
    try:
        from data_collection import get_dataset
        df = get_dataset()
        
        if df is not None:
            print(f"✅ Dataset loaded successfully: {len(df)} messages")
            print(f"   - Spam: {len(df[df['label'] == 'spam'])}")
            print(f"   - Ham: {len(df[df['label'] == 'ham'])}")
            return True
        else:
            print("❌ Dataset loading failed")
            return False
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality"""
    print("\nTesting preprocessing...")
    
    try:
        from data_collection import get_dataset
        from preprocessing import TextPreprocessor
        
        df = get_dataset()
        if df is None:
            print("❌ Cannot test preprocessing without dataset")
            return False
        
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        
        print(f"✅ Preprocessing completed successfully")
        print(f"   - Original columns: {list(df.columns)}")
        print(f"   - Processed columns: {list(df_processed.columns)}")
        
        # Test single message preprocessing
        test_message = "URGENT! You have won a FREE prize! Call NOW!"
        processed = preprocessor.preprocess_text(test_message)
        print(f"   - Test message processed: {processed[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("\nTesting feature extraction...")
    
    try:
        from data_collection import get_dataset
        from preprocessing import TextPreprocessor
        from feature_extraction import prepare_features
        
        df = get_dataset()
        if df is None:
            print("❌ Cannot test feature extraction without dataset")
            return False
        
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        
        # Test feature extraction
        numerical_columns = ['text_length', 'word_count', 'char_count', 'avg_word_length', 
                           'uppercase_count', 'digit_count', 'special_char_count', 'spam_indicators']
        
        X_train, X_test, y_train, y_test, feature_extractor = prepare_features(
            df_processed, numerical_columns=numerical_columns
        )
        
        print(f"✅ Feature extraction completed successfully")
        print(f"   - Training set: {X_train.shape}")
        print(f"   - Testing set: {X_test.shape}")
        print(f"   - Features: {X_train.shape[1]}")
        
        return True
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\nTesting prediction...")
    
    try:
        from prediction import SpamPredictor
        
        predictor = SpamPredictor()
        
        # Test if models exist
        if not predictor.models:
            print("⚠️  No trained models found (this is expected for first run)")
            print("   Run training first to test prediction functionality")
            return True
        
        # Test prediction
        test_message = "Hey there! How are you doing?"
        result = predictor.predict_single(test_message)
        
        print(f"✅ Prediction test completed")
        print(f"   - Test message: {test_message}")
        print(f"   - Prediction: {result['prediction']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("SMS Spam Detection Project - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Collection Test", test_data_collection),
        ("Preprocessing Test", test_preprocessing),
        ("Feature Extraction Test", test_feature_extraction),
        ("Prediction Test", test_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py --mode train' to train models")
        print("2. Run 'python main.py --mode predict --interactive' for predictions")
        print("3. Run 'streamlit run app.py' for web interface")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 