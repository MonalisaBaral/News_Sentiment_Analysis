# Sentiment Analysis for News Headlines

## 📄 Project Overview
This project is a **Sentiment Analysis Web Application** built using **Flask**. The application analyzes the sentiment of news headlines to determine if they are **positive**, **negative**, or **neutral**. The project leverages **Natural Language Processing (NLP)** techniques, including text cleaning, feature extraction, and machine learning models, to provide sentiment predictions.

## 🚀 Features
- User-friendly web interface to input news headlines.
- Sentiment prediction using a pre-trained **Linear Support Vector Regressor (LinearSVR)** model.
- Provides sentiment scores with classifications as **positive**, **negative**, or **neutral**.
- Responsive design for seamless use across devices.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **Machine Learning**: Scikit-Learn, NLTK, TextBlob
- **Data**: TF-IDF Vectorizer, Sentiment Polarity

## 📂 Project Structure
```
sentiment-analysis-app/
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   ├── home.html
│   └── result.html
├── train_file.csv
├── test_file.csv
├── app.py
├── SVRmain.pkl
├── tfidfmain.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies
Make sure you have `pip` installed. Then, run:
```bash
pip install -r requirements.txt
```

### 4. NLTK Setup
Download the necessary NLTK data files:
```python
python -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
"
```

### 5. Running the Flask Application
```bash
python app.py
```
The server will run on `http://127.0.0.1:5000/`. Open this URL in your web browser.

## 📊 How to Use
1. Navigate to the homepage.
2. Enter the **news headline**, **source**, and **topic** in the provided fields.
3. Click on **"Analyze Sentiment"**.
4. The app will display whether the headline is positive, negative, or neutral along with a sentiment score.

## 🧪 Model Training (Optional)
If you want to retrain or modify the model, follow these steps:

1. **Prepare your dataset** (`train_file.csv` and `test_file.csv`):
   - Ensure the data includes columns: `Headline`, `Source`, `Topic`, `SentimentHeadline`.

2. **Run the training script**:
   - Modify and execute the model training code provided in `train_model.py` (if applicable).

3. **Save the model**:
   - Ensure you save the updated model as `SVRmain.pkl` and the TF-IDF vectorizer as `tfidfmain.pkl`.



## 🌐 Deployment
To deploy this app on a cloud platform like **Heroku**:

1. Install the **Heroku CLI** and log in:
    ```bash
    heroku login
    ```

2. Create a `Procfile`:
    ```
    web: python app.py
    ```

3. Commit and push to Heroku:
    ```bash
    git add .
    git commit -m "Deploying sentiment analysis app"
    heroku create
    git push heroku main
    ```

## 🔧 Dependencies
Make sure your `requirements.txt` includes the following packages:
```
Flask==2.3.3
scikit-learn==1.3.1
pandas==2.0.3
numpy==1.24.3
textblob==0.17.1
nltk==3.8.1
scipy==1.11.1
```

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request. Please ensure all new code follows the project's coding guidelines.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📧 Contact
For any inquiries or feedback, please reach out to:
- **Your Name**: monabaral2610@gmail.com
- **GitHub**: [MonalisaBaral](https://github.com/MonalisaBaral)

---
