import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

nltk.download('stopwords')
nltk.download('punkt')

# Load data
train = pd.read_csv('train_file.csv')
test = pd.read_csv('test_file.csv')

# Fill missing values
train['Source'] = train['Source'].fillna('Bloomberg')
test['Source'] = test['Source'].fillna('Bloomberg')

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


# Combine columns and clean text
train['Text_Headline'] = (train['Headline'] + ' ' + train['Source'] + ' ' + train['Topic']).apply(clean)
test['Text_Headline'] = (test['Headline'] + ' ' + test['Source'] + ' ' + test['Topic']).apply(clean)

# Vectorize text
vectorizer = TfidfVectorizer(use_idf=True)
train_v_Headline = vectorizer.fit_transform(train['Text_Headline'])
test_v_Headline = vectorizer.transform(test['Text_Headline'])

# Sentiment analysis
train['polarity_h'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
test['polarity_h'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

train['subjectivity_h'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
test['subjectivity_h'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Label encoding
encoder = LabelEncoder()
train['Topic'] = encoder.fit_transform(train['Topic'])
test['Topic'] = encoder.transform(test['Topic'])

train['Source'] = encoder.fit_transform(train['Source'])
test['Source'] = encoder.transform(test['Source'])

# Feature engineering
train["num_words_h"] = train["Text_Headline"].apply(lambda x: len(str(x).split()))
test["num_words_h"] = test["Text_Headline"].apply(lambda x: len(str(x).split()))

train["num_unique_words_h"] = train["Text_Headline"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words_h"] = test["Text_Headline"].apply(lambda x: len(set(str(x).split())))

train["num_chars_h"] = train["Text_Headline"].apply(lambda x: len(str(x)))
test["num_chars_h"] = test["Text_Headline"].apply(lambda x: len(str(x)))

train["mean_word_len_h"] = train["Text_Headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len_h"] = test["Text_Headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# Scaling
scaler = StandardScaler()
cols = ['num_words_h', 'num_unique_words_h', 'num_chars_h', 'mean_word_len_h', 'polarity_h', 'subjectivity_h']
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

# Combine features
train_X2 = train[cols]
test_X2 = test[cols]

train_X_Headline = hstack([train_v_Headline, csr_matrix(train_X2.values)])
test_X_Headline = hstack([test_v_Headline, csr_matrix(test_X2.values)])
y2 = train['SentimentHeadline']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_X_Headline, y2, test_size=0.20, random_state=42)

# Model training
clf = LinearSVR(C=0.2)
clf.fit(X_train, y_train)

# Save model and vectorizer
with open('SVRmain.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('tfidfmain.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
