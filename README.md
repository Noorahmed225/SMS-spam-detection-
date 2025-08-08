# 📱 SMS Spam Detection Project

A comprehensive machine learning project that builds intelligent models to detect and filter spam SMS messages using Natural Language Processing (NLP) and various machine learning algorithms.

## 🎯 Project Overview

This project implements a complete SMS spam detection system that can:
- **Preprocess** and clean SMS text data
- **Extract features** from text using TF-IDF vectorization
- **Train multiple ML models** (Naive Bayes, Logistic Regression, Random Forest, SVM)
- **Make real-time predictions** on new messages
- **Provide a web interface** for easy interaction
- **Analyze model performance** with detailed metrics

## ✨ Features

- 🔍 **Advanced Text Preprocessing**: URL removal, special character cleaning, lemmatization
- 🤖 **Multiple ML Models**: 4 different algorithms for robust classification
- 📊 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- 🌐 **Web Application**: User-friendly Streamlit interface
- 📈 **Data Visualization**: Performance comparisons and insights
- 🔧 **Easy Setup**: Automated installation and configuration
- 📝 **Interactive Testing**: Command-line and web-based prediction modes

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection

# Install dependencies and setup
python setup.py

# Train the models
python main.py --mode train

# Launch web application
streamlit run app.py
```

## 📁 Project Structure

```
SMS/
├── 📂 data/              # Dataset files
├── 📂 models/            # Trained ML models
├── 📂 notebooks/         # Analysis and EDA scripts
├── 📂 src/               # Core source code
│   ├── data_collection.py    # Dataset loading
│   ├── preprocessing.py      # Text preprocessing
│   ├── feature_extraction.py # Feature engineering
│   ├── model_training.py     # ML model training
│   └── prediction.py         # Prediction functionality
├── 🌐 app.py             # Streamlit web application
├── ⚙️ main.py            # Main execution script
├── 🔧 setup.py           # Project setup script
├── 🧪 test_project.py    # Testing suite
└── 📋 requirements.txt   # Python dependencies
```

## 🛠️ Usage

### Command Line Interface

```bash
# Train models
python main.py --mode train

# Single prediction
python main.py --mode predict --message "URGENT: You've won a prize!"

# Interactive mode
python main.py --mode predict

# Demo mode
python main.py --mode demo
```

### Web Application

```bash
# Launch web interface
streamlit run app.py
```

The web app provides:
- 📝 **Single Message Prediction**: Test individual messages
- 📄 **Batch Prediction**: Upload CSV files with multiple messages
- 📊 **Data Analysis**: Visualize dataset characteristics
- 🎯 **Model Training**: Retrain models with new data
- 📈 **Results**: View model performance metrics

## 🤖 Machine Learning Models

The project implements and compares 4 different algorithms:

| Model | Description | Use Case |
|-------|-------------|----------|
| **Multinomial Naive Bayes** | Probabilistic classifier based on Bayes theorem | Fast, good baseline |
| **Logistic Regression** | Linear model for binary classification | Interpretable, reliable |
| **Random Forest** | Ensemble of decision trees | Robust, handles non-linear patterns |
| **Support Vector Machine** | Kernel-based classifier | High accuracy, complex patterns |

## 📊 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

## 📈 Dataset

The project uses the **SMS Spam Collection Dataset** from UCI Machine Learning Repository:
- **5,574 messages** total
- **747 spam messages** (13.4%)
- **4,827 ham messages** (86.6%)
- **Text preprocessing** applied for feature extraction

## 🔧 Technical Details

### Text Preprocessing Pipeline
1. **Text Cleaning**: Remove URLs, emails, phone numbers
2. **Normalization**: Convert to lowercase, remove special characters
3. **Tokenization**: Split text into words
4. **Stopword Removal**: Remove common words
5. **Lemmatization**: Reduce words to base form

### Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Statistical Features**: Message length, word count, character count
- **Spam Indicators**: Count of suspicious words/patterns

### Model Training
- **Train/Test Split**: 80/20 ratio
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimization
- **Model Persistence**: Save/load trained models

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_project.py
```

Tests cover:
- ✅ Module imports
- ✅ Data collection
- ✅ Text preprocessing
- ✅ Feature extraction
- ✅ Model prediction

## 📦 Dependencies

Key Python packages:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **nltk**: Natural language processing
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plotting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the SMS Spam Collection Dataset
- **Scikit-learn** community for excellent ML tools
- **Streamlit** team for the web framework
- **NLTK** developers for NLP capabilities

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/sms-spam-detection](https://github.com/yourusername/sms-spam-detection)
- **Issues**: [https://github.com/yourusername/sms-spam-detection/issues](https://github.com/yourusername/sms-spam-detection/issues)

## ⭐ Star History

If you find this project helpful, please give it a ⭐ star on GitHub!

---

**Made with ❤️ for the data science community** 