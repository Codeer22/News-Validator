import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from fakenews import read_data, preprocess_text, prepare_data, train_models, evaluate_models

app = Flask(__name__)

# Load the fake and true news data
fake_file = os.path.join('data', 'Fake.csv')
true_file = os.path.join('data', 'True.csv')
df_fake, df_true = read_data(fake_file, true_file)

# Prepare the data
X_train, X_test, Y_train, Y_test, error_message = prepare_data(df_fake, df_true)

if error_message:
    print(error_message)
    # You can choose to exit the application or handle the error in a different way
    exit()

# Train the models
vectorizer, lr, dtc, gclf, rclf = train_models(X_train, Y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        xv_text = vectorizer.transform([processed_text])
        lr_pred = lr.predict(xv_text)[0]
        dtc_pred = dtc.predict(xv_text)[0]
        gclf_pred = gclf.predict(xv_text)[0]
        rclf_pred = rclf.predict(xv_text)[0]
        result = {
            'text': text,
            'lr_pred': 'Fake' if lr_pred == 0 else 'True',
            'dtc_pred': 'Fake' if dtc_pred == 0 else 'True',
            'gclf_pred': 'Fake' if gclf_pred == 0 else 'True',
            'rclf_pred': 'Fake' if rclf_pred == 0 else 'True'
        }
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)