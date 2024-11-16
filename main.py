from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler

# Load necessary NLTK data
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
clf = pickle.load(open('SVRmain.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))


# Load scaler
scaler = StandardScaler()

# Define stop words
stop = set(stopwords.words('english'))


# Clean text function
def clean(text):
    text_token = word_tokenize(text)
    filtered_text = ' '.join([w.lower() for w in text_token if w.lower() not in stop and len(w) > 2])
    filtered_text = re.sub(r"[^a-zA-Z]+", ' ', filtered_text)
    text_only = re.sub(r'\b\d+\b', '', filtered_text)
    clean_text = text_only.replace(',', '').replace('.', '').replace(':', '')
    return clean_text


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['Headline']
        source = request.form['Source']
        topic = request.form['Topic']
        headline_combined = headline + ' ' + source + ' ' + topic
        cleaned_headline = clean(headline_combined)
        transformed_headline = vectorizer.transform([cleaned_headline])

        # Extract features
        polarity_h = TextBlob(str(headline)).sentiment.polarity
        subjectivity_h = TextBlob(str(headline)).sentiment.subjectivity
        num_words_h = len(str(cleaned_headline).split())
        num_unique_words_h = len(set(str(cleaned_headline).split()))
        num_chars_h = len(str(cleaned_headline))
        mean_word_len_h = np.mean([len(w) for w in str(cleaned_headline).split()])

        features = np.array(
            [num_words_h, num_unique_words_h, num_chars_h, mean_word_len_h, polarity_h, subjectivity_h]).reshape(1, -1)
        scaled_features = scaler.fit_transform(features)

        final_features = hstack([transformed_headline, csr_matrix(scaled_features)])

        # Predict sentiment
        my_prediction = clf.predict(final_features)[0]

        return render_template('result.html', my_prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
