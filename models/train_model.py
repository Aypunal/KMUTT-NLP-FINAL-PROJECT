#!/usr/bin/env conda
# -*- coding: utf-8 -*-

"""
train_model.py

Script to:
 - load the phishing emails dataset (Kaggle)
 - minimal text cleaning
 - build multiple pipelines (TF-IDF + classifiers)
 - hyperparameter search (GridSearchCV)
 - compare performances
 - save the best model to disk
"""

import os
import pandas as pd
import numpy as np
import kagglehub


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

import joblib

# --------------------------------------
# 1. Data loading and preparation
# --------------------------------------

def load_data():
    
    path = kagglehub.dataset_download("subhajournal/phishingemails")
    print("Path to dataset files:", path)

    files = os.listdir(path)
    print("Available files:", files)

    csv_files = [f for f in files if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in directory")
    csv_path = os.path.join(path, csv_files[0])
    print("Using file:", csv_path)

    df = pd.read_csv(csv_path)
    return df

def preprocess_text(text_series):
    """
    Minimal text preprocessing:
    - remove line breaks
    - convert to lowercase
    - (optional) remove URLs
    - (optional) remove superfluous punctuation
    """
    # Replace NaN (just in case) with empty string then do dropna upstream
    text_series = text_series.fillna("")

    # Cleaning function
    def clean_once(t):
        # Standardize line breaks
        t = t.replace('\n', ' ')
        # Convert to lowercase
        t = t.lower()
        # Simplified URL removal
        # t = re.sub(r'http[s]?://\S+', ' ', t)
        return t

    return text_series.apply(clean_once)

def prepare_dataset():
    """
    Load, clean and prepare X, y for training.
    """
    df = load_data()

    # 1) Remove rows where 'Email Text' is null
    df = df.dropna(subset=['Email Text']).reset_index(drop=True)

    # 2) Create 'text_clean' column by applying our minimal preprocessing
    df['text_clean'] = preprocess_text(df['Email Text'])

    # 3) Encode target as binary: 'Phishing Email' -> 1, 'Safe Email' -> 0
    df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

    # 4) Return X (text_clean) and y (label)
    X = df['text_clean'].values
    y = df['label'].values

    return X, y

# --------------------------------------
# 2. Pipeline construction and hyperparameter grids
# --------------------------------------

def build_pipelines_and_grids():
    """
    Returns a dictionary of pipelines and a dictionary of hyperparameter grids.
    Key for each model: 'logreg', 'nb', 'rf'
    """
    pipelines = {}
    grids = {}

    # 1) Pipeline + grid for Logistic Regression
    pipelines['logreg'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_df=0.8,
            min_df=5
        )),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])
    grids['logreg'] = {
        'tfidf__max_df': [0.7, 0.8, 0.9],
        'tfidf__min_df': [3, 5, 10],
        'clf__C': [0.01, 0.1, 1, 10]
    }

    # 2) Pipeline + grid for MultinomialNB
    pipelines['nb'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_df=0.8,
            min_df=5
        )),
        ('clf', MultinomialNB())
    ])
    grids['nb'] = {
        'tfidf__max_df': [0.7, 0.8],
        'tfidf__min_df': [3, 5],
        'clf__alpha': [0.1, 0.5, 1.0]
    }

    # 3) Pipeline + grid for RandomForest
    pipelines['rf'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_df=0.8,
            min_df=5
        )),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    grids['rf'] = {
        'tfidf__max_df': [0.7, 0.8],
        'tfidf__min_df': [3, 5],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    return pipelines, grids

# --------------------------------------
# 3. Training, hyperparameter search and comparison
# --------------------------------------

def train_and_evaluate(X, y, pipelines, grids, output_dir='models'):
    """
    For each model defined in pipelines, launch a GridSearchCV,
    display results, then compare performances on test set.
    Finally save the best model in the output_dir directory.
    """
    # 1) Train/test split (80/20), stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Dictionary to store final results
    results = []

    # If output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2) Loop over each pipeline
    for name, pipeline in pipelines.items():
        print(f"\n=== Model: {name} ===")

        grid = grids[name]
        # Instantiate GridSearchCV
        # Use 5 stratified folds on training set
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=5,
            scoring='f1',       # we seek to maximize F1 (precision/recall balance)
            n_jobs=-1,
            verbose=1
        )

        # 3) Launch training (GridSearch)
        gs.fit(X_train, y_train)
        print(f"Best parameters for {name}: {gs.best_params_}")
        print(f"Best CV score (F1) for {name}: {gs.best_score_:.4f}")

        # 4) Evaluation on test set
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        print(f"— TEST Accuracy   : {acc:.4f}")
        print(f"— TEST Precision  : {prec:.4f}")
        print(f"— TEST Recall     : {rec:.4f}")
        print(f"— TEST F1-score   : {f1:.4f}")
        if not np.isnan(auc):
            print(f"— TEST ROC AUC    : {auc:.4f}")

        # 5) Store results in list
        results.append({
            'model': name,
            'best_params': gs.best_params_,
            'cv_f1_mean': gs.best_score_,
            'test_accuracy': acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1': f1,
            'test_auc': auc
        })

        # 6) Save best model
        model_path = os.path.join(output_dir, f"{name}_best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"Model saved as: {model_path}")

    # 7) Compare all results in a DataFrame
    results_df = pd.DataFrame(results).sort_values(by='test_f1', ascending=False)
    print("\n=== Final model comparison (sorted by test_f1) ===")
    print(results_df)
    
    # 8) Select best model based on F1-score
    best_model_name = results_df.iloc[0]['model']
    best_model_path = os.path.join(output_dir, f"{best_model_name}_best_model.joblib")
    best_model = joblib.load(best_model_path)
    
    print("\n=== Best model selected ===")
    print(f"Model: {best_model_name}")
    print(f"F1-score on test: {results_df.iloc[0]['test_f1']:.4f}")
    print(f"Precision on test: {results_df.iloc[0]['test_precision']:.4f}")
    print(f"Recall on test: {results_df.iloc[0]['test_recall']:.4f}")
    print("\n— Classification report —")
    print(classification_report(y_test, y_pred, target_names=["Safe Email", "Phishing Email"]))

    
    # Save best model as final model
    final_model_path = os.path.join(output_dir, "final_model.joblib")
    joblib.dump(best_model, final_model_path)
    print(f"\nBest model saved as: {final_model_path}")

    # Save summary DataFrame
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    return results_df

# --------------------------------------
# 4. Main function
# --------------------------------------

def main():

    # 4.1) Dataset preparation
    print("Dataset preparation (loading + cleaning)...")
    X, y = prepare_dataset()
    print(f"Total number of examples after cleaning: {len(y)}")
    print(f"- Number of safe emails: {(y == 0).sum()}")
    print(f"- Number of phishing emails: {(y == 1).sum()}")

    # 4.2) Pipeline and grid construction
    pipelines, grids = build_pipelines_and_grids()

    # 4.3) Training, hyperparameter search and comparison
    print("Launching training and GridSearchCV for each model...")
    results_df = train_and_evaluate(X, y, pipelines, grids, output_dir='models')
    
    # 4.4) Display final comparison
    print("\nFinal results saved in 'models/model_comparison.csv'.")

if __name__ == "__main__":
    main()

