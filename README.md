<!-- 

1. Create a new environment

```bash
conda env create -f environment.yml
```

2. Activate the environment

```bash
conda activate ai-mail-assistant
```

3. Run the app.py file

```bash
python app.py
``` -->


# 🤖 AI Mail Assistant

This project implements a AI Mail Assistant using classical NLP techniques. The pipeline relies on **lemmatization**, **TTF-IDF** vectorization, and several classification algorithms.


---

## 📊 Overview

- Dataset: Phishing Emails
- Task: Spam Detection & Generation of a response
<!-- - Models: Logistic Regression, Multinomial Naive Bayes, SGD Classifier, Hugging Face -->
- Evaluation: Accuracy, Precision, Recall, F1-score
- Tools: Python, Scikit-learn, Pandas, NumPy
- Environment: Conda

---

Quick start with :

1. Clone the repository

```bash
git clone https://github.com/Aypunal/KMUTT-NLP-FINAL-PROJECT.git
```

2. Create the environment:

```bash
conda env create -f environment.yml
conda activate ai-mail-assistant
```

3. Train the model

```bash
python model/train_model.py
```

4. Run the app.py file

```bash
python app.py
```

## 📂 Dataset

- **Source**: [`subhajournal/phishingemails`](https://www.kaggle.com/datasets/subhajournal/phishingemails/)
- Accessed via [`kagglehub`](https://pypi.org/project/kagglehub/)
- Format: `.csv` file with labels (`Safe Email`, `Phishing Email`)
---

<!-- ## ⚙️ Pipeline Steps

1. **Dataset Loading**
   - Load 18,600 rows from `emails.csv`
   - Parse labels and text content

2. **Text Preprocessing**
   - Remove punctuation
   - Lowercase
   - Lemmatize words (NLTK WordNetLemmatizer, no POS tagging)
   - Remove English stopwords (via CountVectorizer)
   - Extract 1-grams and 2-grams
   - Limit dictionary to top 3000 tokens (`max_features=3000`)

3. **Vectorization**
   - `CountVectorizer` used to convert text into sparse matrix (BoW) (alternatively, `TfidfVectorizer` can be used or https://fasttext.cc/ for embeddings) 
   - Output matrix shape: `(10000, 3000)`

4. **Train/Test Split**
   - 80% training / 20% testing using `train_test_split`

5. **Model Training & Evaluation**
   - Models tested:
     - Logistic Regression
     - Multinomial Naive Bayes
     - SGD Classifier
   - Metrics:
     - Accuracy
     - Precision / Recall / F1-score (`classification_report`)

6. **Custom Prediction**
   - Run `predict_sentiment(comment, model, model_name)`
   - Test your own comment (e.g., `"This product was terrible"`)

---

## 🧪 Example Test Comments

```python
predict_sentiment("Absolutely love it, highly recommend this product.", model=clf, model_name="Logistic Regression")
predict_sentiment("This product was terrible and broke in two days.", model=clf, model_name="Logistic Regression")
predict_sentiment("The quality is decent, but the price is too high.", model=clf, model_name="Logistic Regression")
```

--- -->

## 📝 Requirements

See [environment.yml](./environment.yml)

<!-- ---

The full explanation, methodology, model comparison, and discussion of performance and limitations are available in [report.pdf](./67540460063_RomainBLANCHOT_report.pdf).

---  -->

## 📜 License

[GNU GENERAL PUBLIC LICENSE](./LICENSE)

--- 