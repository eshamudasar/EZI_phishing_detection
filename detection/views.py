import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
import joblib
from django.http import HttpResponse
from django.conf import settings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Paths for your models, vectorizers, and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_ham_detection_model(url).h5')
MODEL_ENG_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_ham_detection_model-eng.h5')
MODEL_URDU_PATH = os.path.join(BASE_DIR, 'FYP', 'spam_detection_model_urdu.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'FYP', 'standard_scaler.pkl')
TFIDF_ENG_PATH = os.path.join(BASE_DIR, 'FYP', 'tfidf_vectorizer_eng.pkl')
TFIDF_URDU_PATH = os.path.join(BASE_DIR, 'FYP', 'tfidf_vectorizer_urdu.pkl')

# Load models and vectorizers
phishing_model = tf.keras.models.load_model(MODEL_URL_PATH)
eng_model = tf.keras.models.load_model(MODEL_ENG_PATH)
urdu_model = tf.keras.models.load_model(MODEL_URDU_PATH)
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
    return "phishing" if prediction[0][0] >= 0.5 else "safe"

def detect_language(text):
    if any(char in text for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return 'english'
    elif any(char in text for char in 'ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ہ ھ ی ے'):
        return 'urdu'
    else:
        return 'unknown'

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

# Real-time Phishing Detection View with Web Scraping and Download Button
def real_time_phishing(request):
    file_name = None
    result = None
    file_url = None
    page_text = None
    image_url = None

    if request.method == 'POST':
        url = request.POST.get('url')

        # Set up Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode (no GUI)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        try:
            # Open the URL in the browser and scrape the content
            driver.get(url)
            time.sleep(3)  # Give the page some time to load
            
            # Get page content using BeautifulSoup
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Scrape the text content from the page
            page_text = soup.get_text()

            # Detect language of the extracted text
            language = detect_language(page_text)

            # Classify each line of text as "spam" or "ham"
            lines = page_text.split('\n')
            labeled_lines = []
            for line in lines:
                if line.strip():  # Ignore empty lines
                    if language == 'english':
                        label = detect_phishing(line, 'english')
                    elif language == 'urdu':
                        label = detect_phishing(line, 'urdu')
                    else:
                        label = 'unknown'
                    labeled_lines.append(f"{label}: {line}")

            # Save the scraped content to a temporary text file with labels
            file_name = f"scraped_data_{int(time.time())}.txt"
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(labeled_lines))


            # Store the file URL for the download button
            file_url = file_name

            # Classify the URL using phishing detection logic
            result = predict_phishing(url)
            result = f"The entered URL is classified as {result}."
            image_url = 'images/giphy.webp' if "phishing" in result else 'images/safe.webp'
        except Exception as e:
            result = f"An error occurred while scraping the URL: {str(e)}"
        
        finally:
            driver.quit()

    # Render the result with a download button if the file is ready
    return render(request, 'real_time_phishing.html', {'result': result, 'file_url': file_url, 'image_url': image_url})


# Download the scraped file when the user clicks the button
def download_file(request, file_name):
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='text/plain')
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            return response
    else:
        return HttpResponse("File not found.", status=404)

# Views for other sections of the website remain the same
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def dataset(request):
    folder_path = os.path.join(settings.MEDIA_ROOT, 'FYP')  # Corrected path to MEDIA_ROOT
    files = []
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
    return render(request, 'dataset.html', {'files': files, 'folder_path': folder_path})

def downloads(request):
    downloads = [
        {'name': 'Phishing Detection User Guide', 'url': '#'},
        {'name': 'Datasets for Phishing Detection', 'url': 'dataset.html'},
    ]
    return render(request, 'downloads.html', {'downloads': downloads})

def login(request):
    return render(request, 'login.html')

def phishintro(request):
    return render(request, 'phishintro.html', {})

def contact(request):
    return render(request, 'contact.html')

def keyword_phishing(request):
    result = None
    image_url = None

    if request.method == 'POST':
        text = request.POST.get('text')
        language = detect_language(text)
        if language == 'unknown':
            result = "Unsupported language"
        else:
            result = detect_phishing(text, language)
            result = f"The entered text is classified as {result}."
            image_url = 'images/scam.webp' if "spam" in result else 'images/legit.webp'
    
    return render(request, 'keyword_phishing.html', {'result': result, 'image_url': image_url})

def url_phishing(request):
    result = None
    image_url = None
    if request.method == 'POST':
        url = request.POST.get('url')
        result = predict_phishing(url)
        result = f"The entered URL is classified as {result}."
        image_url = 'images/giphy.webp' if "phishing" in result else 'images/safe.webp'
    return render(request, 'url_phishing.html', {'result': result, 'image_url': image_url})
