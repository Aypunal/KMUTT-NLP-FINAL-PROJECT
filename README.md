# 🤖 AI Mail Assistant

📌 **Table of Contents**
- [📝 Overview](#-overview)
- [🚀 Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [⚙️ Installation](#️-installation)
- [▶️ Usage](#️-usage)
- [📁 Project Structure](#-project-structure)
- [✉️ Email Examples](#️-email-examples)
- [⚠️ Limitations](#️-limitations)
- [📈 Future Improvements](#-future-improvements)
- [👨‍💻 Authors & Credits](#-authors--credits)

---

## 📝 Overview

The **AI Mail Assistant** is an intelligent email classification system that automatically detects spam and phishing emails using classical NLP techniques. The system combines **TF-IDF vectorization** with machine learning classifiers to provide accurate email classification and includes a web-based interface for real-time email analysis.

- **Dataset**: Phishing Emails from Kaggle ([`subhajournal/phishingemails`](https://www.kaggle.com/datasets/subhajournal/phishingemails/))
- **Task**: Binary classification (Safe Email vs Phishing Email) + Response generation
- **Best Model**: Logistic Regression (F1-score: 0.9657, Accuracy: 97.26%)
- **Interface**: Flask web application with modern UI

---

## 🚀 Features

- ✅ **High-Accuracy Spam Detection**: 97.26% accuracy using optimized Logistic Regression
- 🔍 **Real-time Email Analysis**: Web interface for instant email classification
- 📊 **Multiple Model Comparison**: Automated training and evaluation of Logistic Regression, Naive Bayes, and Random Forest
- 🤖 **Smart Response Generation**: AI-powered response suggestions for legitimate emails using Transformers
- 📈 **Comprehensive Evaluation**: Detailed performance metrics (Precision, Recall, F1-score, ROC-AUC)
- 🛡️ **Robust Preprocessing**: Advanced text cleaning and TF-IDF vectorization pipeline
- 💾 **Model Persistence**: Automated model saving and loading for production use

---

## 🛠️ Technologies Used

### **Machine Learning & NLP**
- **Scikit-learn**: TF-IDF vectorization, classification algorithms, model evaluation
- **Pandas & NumPy**: Data manipulation and numerical computations
- **NLTK**: Text preprocessing and tokenization

### **Deep Learning & Transformers**
- **Transformers (Hugging Face)**: Response generation using FLAN-T5
- **PyTorch**: Backend for transformer models

### **Web Framework**
- **Flask**: Web application framework
- **HTML/CSS**: Modern, responsive user interface

### **Data & Environment**
- **KaggleHub**: Dataset downloading and management
- **Conda**: Environment and dependency management
- **Joblib**: Model serialization and persistence

---

## ⚙️ Installation

### **Prerequisites**
- Conda or Miniconda installed
- Python 3.12+

### **Step-by-step Installation**

1. **Clone the repository**
```bash
git clone https://github.com/Aypunal/KMUTT-NLP-FINAL-PROJECT.git
cd KMUTT-NLP-FINAL-PROJECT
```

2. **Create and activate the conda environment**
```bash
conda env create -f environment.yml
conda activate ai-mail-assistant
```

3. **Train the model** (optional - pre-trained model included)
```bash
python models/train_model.py
```

4. **Run the web application**
```bash
python app.py
```

5. **Access the application**
   - Open your browser and navigate to `http://127.0.0.1:5000`

---

## ▶️ Usage

### **Web Interface**
1. Launch the Flask application: `python app.py`
2. Open `http://127.0.0.1:5000` in your browser
3. Paste the email content into the text area
4. Click "Analyse" to get instant classification results
5. For legitimate emails, receive AI-generated response suggestions

### **Model Training**
```bash
# Train all models and compare performances
python models/train_model.py

# Results will be saved in:
# - models/logreg_best_model.joblib (best performing model)
# - models/model_comparison.csv (performance comparison)
```

### **API Usage** (Programmatic)
```python
import joblib

# Load the trained model
model = joblib.load("models/logreg_best_model.joblib")

# Classify an email
email_text = "Your email content here..."
prediction = model.predict([email_text])[0]
# 0 = Safe Email, 1 = Phishing Email
```

---

## 📁 Project Structure

```
KMUTT-NLP-FINAL-PROJECT/
├── 📄 README.md                    # Project documentation
├── 📄 LICENSE                      # GNU GPL License
├── 📄 environment.yml              # Conda environment configuration
├── 📄 app.py                       # Flask web application
├── 📁 models/
│   ├── 📄 train_model.py          # Model training script
│   ├── 📄 model_comparison.csv     # Performance comparison results
│   ├── 📄 logreg_best_model.joblib # Best trained model (Logistic Regression)
│   ├── 📄 rf_best_model.joblib     # Random Forest model
│   └── 📄 nb_best_model.joblib     # Naive Bayes model
└── 📁 templates/
    └── 📄 index.html               # Web interface template
```

---

## ✉️ Email Examples

### **Safe Email Example**
```
Subject: Meeting Reminder - Project Update

Hi Team,

This is a reminder about our project update meeting scheduled for tomorrow at 2 PM.
Please prepare your status reports and bring any questions you might have.

Best regards,
John
```
**Result**: ✅ Safe Email + AI response suggestion

### **Phishing Email Example**
```
URGENT: Your account will be suspended!

Dear Customer,
Your PayPal account has been limited. Click here immediately to verify:
http://fake-paypal-site.com/verify

Enter your login credentials to restore access.
```
**Result**: 🚫 SPAM - Delete immediately

---

## ⚠️ Limitations

- **Language Support**: Currently optimized for English emails only
- **Model Scope**: Trained specifically on phishing/spam detection (binary classification)
- **Response Generation**: AI responses are suggestions and may require human review
- **Dataset Bias**: Performance may vary on email types not well-represented in training data
- **Real-time Updates**: Model requires retraining to adapt to new spam techniques

---

## 📈 Future Improvements

- 🌐 **Multi-language Support**: Extend classification to French, Spanish, and other languages
- 🧠 **Deep Learning Integration**: Implement BERT/RoBERTa for improved context understanding
- 📱 **Mobile Application**: Develop iOS/Android apps for on-the-go email analysis
- 🔄 **Continuous Learning**: Implement online learning for model adaptation
- 📧 **Email Client Integration**: Browser extensions for Gmail, Outlook, etc.
- 🎯 **Advanced Classification**: Multi-class classification (spam types, priorities, etc.)
- 🔐 **Privacy Features**: On-device processing for sensitive emails

---

## 👨‍💻 Authors & Credits

**Developed by**: [Romain](https://github.com/LeMarechalDeFer) & [Victor](https://github.com/Aypunal)  
**Course**: KMUTT NLP Final Project  
**Institution**: King Mongkut's University of Technology Thonburi  

### **Acknowledgments**
- Dataset: [`subhajournal/phishingemails`](https://www.kaggle.com/datasets/subhajournal/phishingemails/) on Kaggle
- Transformers: Hugging Face team for FLAN-T5 model
- UI Inspiration: Modern web design principles

### **License**
This project is licensed under the [GNU General Public License v3.0](./LICENSE)

---

**⭐ If this project helps you, please give it a star on GitHub!**

## 📜 License

[GNU GENERAL PUBLIC LICENSE](./LICENSE)

--- 