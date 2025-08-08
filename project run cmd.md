# SMS Spam Detection Project - Commands Guide

## **Step-by-Step Commands to Run the Project**

### **1. Initial Setup (First time only)**
```bash
# Install all dependencies and setup environment
python setup.py
```

### **2. Verify Installation**
```bash
# Test all modules are working correctly
python test_project.py
```

### **3. Train the Models**
```bash
# Train all machine learning models (Naive Bayes, Logistic Regression, Random Forest, SVM)
python main.py --mode train
```

### **4. Make Predictions**
```bash
# Single message prediction - Spam example
python main.py --mode predict --message "URGENT: You've won a prize! Click here to claim"

# Single message prediction - Normal message example
python main.py --mode predict --message "Hi, can we meet tomorrow at 3pm?"

# Interactive prediction mode (enter messages one by one)
python main.py --mode predict
```

### **5. Run Demo Mode**
```bash
# Run demo with sample predictions
python main.py --mode demo
```

### **6. Launch Web Application**
```bash
# Start the Streamlit web app (opens in browser)
streamlit run app.py
```

### **7. Run Exploratory Data Analysis**
```bash
# Run EDA script to analyze dataset
python notebooks/01_EDA.py
```

## **Quick Start Commands (Recommended Order)**

If you want to get started immediately, run these commands in order:

```bash
# 1. Setup (if not done already)
python setup.py

# 2. Train models
python main.py --mode train

# 3. Launch web app
streamlit run app.py
```

## **What Each Command Does:**

- **`python setup.py`**: Installs dependencies, downloads NLTK data, creates directories
- **`python test_project.py`**: Verifies all modules work correctly
- **`python main.py --mode train`**: Trains 4 ML models (Naive Bayes, Logistic Regression, Random Forest, SVM)
- **`python main.py --mode predict`**: Makes predictions on new messages
- **`python main.py --mode demo`**: Shows sample predictions
- **`streamlit run app.py`**: Opens web interface for easy interaction
- **`python notebooks/01_EDA.py`**: Analyzes the dataset and creates visualizations

## **Expected Output:**
- After training, you'll see model performance metrics
- Models will be saved in the `models/` directory
- The web app will open in your browser at `http://localhost:8501`

## **Troubleshooting Commands:**

If you encounter issues:

```bash
# Check if all dependencies are installed
pip list

# Reinstall dependencies
pip install -r requirements.txt

# Download NLTK data manually
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Check Python version
python --version
```

## **Project Structure:**
```
SMS/
├── data/              # Dataset files
├── models/            # Trained models
├── notebooks/         # Analysis scripts
├── src/               # Source code
├── app.py            # Web application
├── main.py           # Main script
├── setup.py          # Setup script
├── test_project.py   # Test script
└── requirements.txt  # Dependencies
```

## **Usage Examples:**

### **Example 1: Complete Training and Web App**
```bash
python setup.py
python main.py --mode train
streamlit run app.py
```

### **Example 2: Quick Prediction Test**
```bash
python main.py --mode predict --message "Free entry in 2 a wkly comp to win FA Cup final tkts"
```

### **Example 3: Interactive Mode**
```bash
python main.py --mode predict
# Then type messages when prompted
```

---

**Note**: Make sure you're in the project directory (`C:\PYTHON\SMS`) when running these commands.
