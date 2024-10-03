import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import joblib
from django.http import HttpResponse

# Paths for your models, vectorizers, and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_ham_detection_model(url).h5')
MODEL_ENG_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_ham_detection_model-eng.h5')
MODEL_URDU_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_detection_model_urdu.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'FYP', 'standard_scaler.pkl')
TFIDF_ENG_PATH = os.path.join(BASE_DIR, 'FYP', 'tfidf_vectorizer_eng.pkl')
TFIDF_URDU_PATH = os.path.join(BASE_DIR, 'FYP', 'tfidf_vectorizer_urdu.pkl')

# Load models and vectorizers
phishing_model = load_model(MODEL_URL_PATH)
eng_model = load_model(MODEL_ENG_PATH)
urdu_model = load_model(MODEL_URDU_PATH)
scaler = joblib.load(SCALER_PATH)
eng_vectorizer = joblib.load(TFIDF_ENG_PATH)
urdu_vectorizer = joblib.load(TFIDF_URDU_PATH)


# Function to extract features from a URL for URL Phishing detection
def extract_features(url):
    parsed_url = urlparse(url)
    url_length = len(url)
    domain_length = len(parsed_url.netloc)
    uses_https = 1 if parsed_url.scheme == 'https' else 0
    suspicious_keywords = 0  # Adjust logic as needed
    return [url_length, domain_length, uses_https, suspicious_keywords]


# URL Phishing Detection
def predict_phishing(url):
    features = np.array([extract_features(url)])
    features_scaled = scaler.transform(features)
    prediction = phishing_model.predict(features_scaled)
    print(prediction)
    return "phishing" if prediction[0][0] >= 0.5 else "safe"

# Text Phishing Detection for English and Urdu
def detect_phishing(text, language):
    if language == 'english':
        preprocessed_text = preprocess_text(text)
        tfidf_vector = eng_vectorizer.transform([preprocessed_text])
        print(tfidf_vector.toarray())
        prediction = eng_model.predict(tfidf_vector.toarray())
        return "spam" if prediction[0] > 0.5 else "ham"
    elif language == 'urdu':
        preprocessed_text = preprocess_text(text)
        tfidf_vector = urdu_vectorizer.transform([preprocessed_text])
        print(tfidf_vector.toarray())
        prediction = urdu_model.predict(tfidf_vector.toarray())
        return "spam" if prediction[0] > 0.5 else "ham"
    

# Text preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text

# Determine language of text
def detect_language(text):
    if any(char in text for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return 'english'
    elif any(char in text for char in 'ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ہ ھ ی ے'):
        return 'urdu'
    else:
        return 'unknown'

# Views for each section of the website

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def downloads(request):
    return render(request, 'downloads.html')

# Keyword Phishing Detection View
def keyword_phishing(request):
    result = None
    if request.method == 'POST':
        text = request.POST.get('text')
        language = detect_language(text)
        if language == 'unknown':
            result = "Unsupported language"
        else:
            result = detect_phishing(text, language)
            result = f"The entered text is classified as {result}."
    return render(request, 'keyword_phishing.html', {'result': result})

# URL Phishing Detection View
def url_phishing(request):
    result = None
    if request.method == 'POST':
        url = request.POST.get('url')
        result = predict_phishing(url)
        result = f"The entered URL is classified as {result}."
    return render(request, 'url_phishing.html', {'result': result})

# Real-time Phishing Detection View
def real_time_phishing(request):
    result = None
    if request.method == 'POST':
        url = request.POST.get('url')
        result = predict_phishing(url)
        result = f"The entered URL is classified as {result}."
    return render(request, 'real_time_phishing.html', {'result': result})
