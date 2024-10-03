import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
import joblib

def extract_features(url):
    parsed_url = urlparse(url)
    url_length = len(url)
    domain_length = len(parsed_url.netloc)
    uses_https = 1 if parsed_url.scheme == 'https' else 0
    suspicious_keywords = 0  # You can implement your own logic to count suspicious keywords
    return [url_length, domain_length, uses_https, suspicious_keywords]

def predict_phishing(url, scaler, model):
    features = np.array([extract_features(url)])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0][0]

# Load the trained model
model = load_model("spam_ham_detection_model(url).h5")

# Load the scaler
scaler = joblib.load("standard_scaler.pkl")

# Get input URL from user
url = input("Enter the URL to check: ")

# Parse the URL
parsed_url = urlparse(url)

# Check if the URL scheme is valid
if parsed_url.scheme not in ['http', 'https']:
    print(f"{url} has an unknown scheme. Skipping prediction.")
else:
    # Make prediction and output result
    prediction = predict_phishing(url, scaler, model)
    if prediction >= 0.5:
        print(f"{url} is predicted to be a phishing URL.")
    else:
        print(f"{url} is predicted to be a safe URL.")
        
