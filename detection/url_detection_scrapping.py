import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
import joblib
import pygetwindow as gw
import pyperclip
import re
import pyautogui
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import signal

last_checked_url = ""

def extract_features(url):
    parsed_url = urlparse(url)
    url_length = len(url)
    domain_length = len(parsed_url.netloc)
    uses_https = 1 if parsed_url.scheme == 'https' else 0
    suspicious_keywords = 0  
    return [url_length, domain_length, uses_https, suspicious_keywords]

def predict_phishing(url, scaler, model):
    features = np.array([extract_features(url)])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0][0]

def scrape_text_from_url(driver):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text

def detect_phishing_eng(text, text_model, text_vectorizer):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = text_vectorizer.transform([preprocessed_text])
    tfidf_vector_dense = tfidf_vector.toarray()
    prediction = text_model.predict(tfidf_vector_dense)
    return "spam" if prediction > 0.5 else "ham"

def detect_phishing_urdu(text, text_model, text_vectorizer):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = text_vectorizer.transform([preprocessed_text])
    tfidf_vector_dense = tfidf_vector.toarray()
    prediction = text_model.predict(tfidf_vector_dense)
    return "spam" if prediction > 0.5 else "ham"

def detect_language(text):
    if any(char in text for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return 'english'
    elif any(char in text for char in 'ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ہ ھ ی ے'):
        return 'urdu'
    else:
        return 'unknown'

# Load model and scaler for URL phishing detection
phishing_model = load_model("spam_ham_detection_model(url).h5")
scaler = joblib.load("standard_scaler.pkl")

# Load model and vectorizer for text spam detection (English)
eng_model = load_model("spam_ham_detection_model-eng.h5")
eng_vectorizer = joblib.load("tfidf_vectorizer_eng.pkl")

# Load model and vectorizer for text spam detection (Urdu)
urdu_model = load_model("spam_detection_model_urdu.h5")
urdu_vectorizer_file = "tfidf_vectorizer_urdu.pkl"

if os.path.exists(urdu_vectorizer_file):
    urdu_vectorizer = joblib.load(urdu_vectorizer_file)
else:
    print(f"Error: {urdu_vectorizer_file} not found.")
    urdu_vectorizer = None

# Selenium setup
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: Run in headless mode
chrome_service = Service("chromedriver.exe")  # Update with the path to your updated chromedriver

# Start the browser
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

def get_active_url():
    global last_checked_url
    
    windows = gw.getWindowsWithTitle(' - Google Chrome')
    if not windows:
        return None
    
    window = windows[0]
    window.maximize()  # Maximize the window
    time.sleep(0.1)
    
    # Simulate Ctrl+L (focus on the address bar) and Ctrl+C (copy the URL)
    pyautogui.hotkey('ctrl', 'l')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)

    # Get the URL from the clipboard
    url = pyperclip.paste()
    if re.match(r'^https?:\/\/', url) and url != last_checked_url:
        last_checked_url = url
        return url
    
    return None

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    driver.quit()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# File to save URL and content labels
output_file = "labeled_urls_and_content.txt"

try:
    while True:
        url = get_active_url()
        if url:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ['http', 'https']:
                print(f"{url} has an unknown scheme. Skipping prediction.")
            else:
                print(f"Checking URL: {url}")
                driver.get(url)
                time.sleep(2)  # Let the page load (adjust as necessary)

                # Scrape text from URL
                text = scrape_text_from_url(driver)
                if text.strip() == '':
                    print(f"No text content found at {url}")
                else:
                    # Detect language of the extracted text
                    language = detect_language(text)
                    print("Detected language:", language)

                    content_labels = []
                    lines = text.split('\n')
                    for line in lines:
                        if line.strip():  # Ignore empty lines
                            if language == 'english':
                                spam_prediction = detect_phishing_eng(line, eng_model, eng_vectorizer)
                                label = "Spam, English" if spam_prediction == "spam" else "Ham, English"
                            elif language == 'urdu' and urdu_vectorizer is not None:
                                spam_prediction = detect_phishing_urdu(line, urdu_model, urdu_vectorizer)
                                label = "Spam, Urdu" if spam_prediction == "spam" else "Ham, Urdu"
                            else:
                                label = "Unknown language"

                            content_labels.append(label + ": " + line)
                            print(label + ": " + line)
                    
                    # Save the URL label and content label to the file
                    url_label = "phishing" if predict_phishing(url, scaler, phishing_model) >= 0.5 else "safe"
                    if content_labels:
                        # Open the file in append mode
                        with open(output_file, "a", encoding='utf-8') as file:
                            # Write the URL and its predicted label
                            file.write(f"{url} is predicted to be a {url_label} URL.\n")
         
                            # Write content labels
                            for content_label in content_labels:
                                file.write(f"Content Label: {content_label}\n")
        
                            # Add a newline to separate entries
                            file.write("\n")
                    else:
                        # Open the file in append mode
                        with open(output_file, "a", encoding='utf-8') as file:
                            # Write the URL and its predicted label
                            file.write(f"{url} is predicted to be a {url_label} URL.\n")
                            # Add a newline to separate entries
                            file.write("\n")

        time.sleep(5)  # Adjust the delay as necessary

        # Check if the Chrome browser window is closed
        if not gw.getWindowsWithTitle(' - Google Chrome'):
            break  # Exit the loop if the browser window is closed

finally:
    driver.quit()
