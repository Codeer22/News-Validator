import pandas as pd
import numpy as np
import re
import nltk  # Assuming you have nltk installed (e.g., pip install nltk)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def read_data(fake_file, true_file):
    """Reads fake and true news data from CSV files."""
    try:
        df_fake = pd.read_csv('Fake.csv')
        df_true = pd.read_csv('True.csv')
        return df_fake, df_true
    except FileNotFoundError:
        print(f"Error: CSV files '{fake_file}' or '{true_file}' not found.")
        return None, None

def preprocess_text(text):
    """Preprocesses text for machine learning."""
    text = text.lower()
    text = re.sub(r'[()]', '', text)  # Remove parentheses
    text = re.sub(r'\W', ' ', text)  # Replace non-alphanumeric with spaces
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^\s\d\w]', '', text)  # Remove punctuation (except for digits and underscores)
    text = re.sub(r'\n', '', text)  # Remove newlines
    # Consider a more nuanced approach for handling alphanumeric text with digits (e.g., keeping relevant terms)
    return text

def prepare_data(df_fake, df_true):
    """Prepares data for machine learning."""
    if df_fake is None or df_true is None:
        return None, None, None, None, "Error: Failed to load CSV files."

    df_fake["target"] = 0
    df_true["target"] = 1
    df = pd.concat([df_fake, df_true], axis=0)
    df = df.sample(frac=1, random_state=42)  # Shuffle data with random state for reproducibility
    df.reset_index(inplace=True, drop=True)  # Reset index
    df["text"] = df["text"].apply(preprocess_text)
    X = df["text"]
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return X_train, X_test, Y_train, Y_test, None

def train_models(X_train, Y_train):
    """Trains machine learning models."""
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(X_train)

    lr = LogisticRegression()
    lr.fit(xv_train, Y_train)

    dtc = DecisionTreeClassifier()
    dtc.fit(xv_train, Y_train)

    gclf = GradientBoostingClassifier()
    gclf.fit(xv_train, Y_train)

    rclf = RandomForestClassifier()
    rclf.fit(xv_train, Y_train)

    return vectorizer, lr, dtc, gclf, rclf

def evaluate_models(X_test, Y_test, vectorizer, lr, dtc, gclf, rclf):
    """Evaluates machine learning models."""
    xv_test = vectorizer.transform(X_test)

    lr_pred = lr.predict(xv_test)
    dtc_pred = dtc.predict(xv_test)
    gclf_pred = gclf.predict(xv_test)
    rclf_pred = rclf.predict(xv_test)

    print("Logistic Regression Accuracy:", accuracy_score(Y_test, lr_pred))
    print("Decision Tree Classifier Accuracy:", accuracy_score(Y_test, dtc_pred))
    print("Gradient Boosting Classifier Accuracy:", accuracy_score(Y_test, gclf_pred))
    print("Random Forest Classifier Accuracy:", accuracy_score(Y_test, rclf_pred))
