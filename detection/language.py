import pandas as pd
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
import os

# Load the English phishing detection model and TF-IDF vectorizer
eng_model = load_model("spam_ham_detection_model.h5")
eng_vectorizer_file = "tfidf_vectorizer.pkl"
if os.path.exists(eng_vectorizer_file):
    eng_vectorizer = joblib.load(eng_vectorizer_file)
else:
    print(f"Error: {eng_vectorizer_file} not found.")
    exit()

# Load the Urdu phishing detection model and TF-IDF vectorizer if available
urdu_model = load_model("spam_detection_model_urdu.h5")
urdu_vectorizer_file = "tfidf_vectorizer_urdu.pkl"
if os.path.exists(urdu_vectorizer_file):
    urdu_vectorizer = joblib.load(urdu_vectorizer_file)
else:
    print(f"Error: {urdu_vectorizer_file} not found.")
    urdu_vectorizer = None

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text

def detect_phishing_eng(text):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = eng_vectorizer.transform([preprocessed_text])
    tfidf_vector_dense = tfidf_vector.toarray()
    prediction = eng_model.predict(tfidf_vector_dense)
    return "spam" if prediction > 0.5 else "ham"

def detect_phishing_urdu(text):
    if urdu_vectorizer is None:
        print("Urdu phishing detection is not available.")
        return None
    
    preprocessed_text = preprocess_text(text)
    tfidf_vector = urdu_vectorizer.transform([preprocessed_text])
    tfidf_vector_dense = tfidf_vector.toarray()
    
    prediction = urdu_model.predict(tfidf_vector_dense)
    return "spam" if prediction > 0.5 else "ham"

def detect_language(text):
    # You can use any language detection library here
    # For demonstration, let's assume we have a simple rule-based approach
    if any(char in text for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return 'english'
    elif any(char in text for char in 'ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ہ ھ ی ے'):
        return 'urdu'
    else:
        return 'unknown'

def main():
    input_text = input("Enter text: ")
    language = detect_language(input_text)
    print("Detected language:", language)

    if language == 'english':
        phishing_result = detect_phishing_eng(input_text)
        print("Phishing detected in English text:", phishing_result)
    elif language == 'urdu':
        phishing_result = detect_phishing_urdu(input_text)
        if phishing_result is not None:
            print("Phishing detected in Urdu text:", phishing_result)
    else:
        print("Unsupported language.")

if __name__ == "__main__":
    main()
