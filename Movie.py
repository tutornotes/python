import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import joblib
import os
import sys


DATASET_PATH = 'Dataset/IMDB.csv'
MODELS_DIR = 'Models/'


os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


print("Downloading NLTK resources (stopwords, punkt)...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}. Please check connection.")
    sys.exit(1)
print("NLTK downloads complete.\n")



def create_sample_csv(path):
    """Creates a small sample IMDB.csv if the actual file is not found."""
    if os.path.exists(path):
        return

    print(f"** {os.path.basename(path)} not found. Creating a sample file for demonstration. **")
    sample_data = {
        'review': [
            "A truly wonderful and brilliant movie. The acting was superb and the plot was engaging. I loved it! <br />",
            "This film was a complete waste of time. Utterly boring and poorly directed. Terrible.",
            "I liked the movie, it was a positive experience.",
            "Such a disaster, I couldn't finish it. The worst.",
            "The best movie I have seen this year, highly recommended.",
            "Absolutely horrible acting and a dreadful script. Do not watch this.",
            "A decent effort, somewhat positive.",
            "Completely negative and uninspired movie experience.",
            "The movie was fun and a real surprise package!",
            "It's a huge disappointment and a failure in every sense."
        ],
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive', 
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(path, index=False)
    print(f"Sample data saved to {path}. For production use, replace this with the full dataset.\n")

create_sample_csv(DATASET_PATH)


def clean_html(text):
    """1. Remove HTML tags"""
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

def is_special(text):
    """2. Remove special characters (keep alphanumeric)"""
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    """3. Convert everything to lowercase"""
    return text.lower()

def rem_stopwords(text):
    """4. Remove stopwords (returns list of words)"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_text(text):
    """5. Stemming (takes list of words, returns single string)"""
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])




def run_sentiment_analysis():
    print("# Movie Review Sentiment Analysis System")
    print("-" * 50)
    
    
    try:
        dataset = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Dataset file not found at {DATASET_PATH}. Please ensure the file is present.")
        return

    print(f"Dataset shape : {dataset.shape}\n")

    
    dataset.sentiment.replace('positive', 1, inplace=True)
    dataset.sentiment.replace('negative', 0, inplace=True)
    print(f"Sentiment encoded ('positive': 1, 'negative': 0)\n")

    
    print("--- Starting Text Cleaning ---")
    dataset.review = dataset.review.apply(clean_html)
    dataset.review = dataset.review.apply(is_special)
    dataset.review = dataset.review.apply(to_lower)
    dataset.review = dataset.review.apply(rem_stopwords)
    dataset.review = dataset.review.apply(stem_text)
    print("--- Text Cleaning Complete ---\n")

    
    initial_shape = dataset.shape[0]
    dataset = dataset[dataset.review.str.strip().astype(bool)]
    print(f"Filtered: {initial_shape - dataset.shape[0]} rows removed due to empty review after cleaning.")
    
    X = np.array(dataset.review.values) # Use the cleaned review column
    y = np.array(dataset.sentiment.values)
    
    cv = CountVectorizer(max_features = 2000) 
    X = cv.fit_transform(dataset.review).toarray()
    
  
    joblib.dump(cv, os.path.join(MODELS_DIR, "MRSA_CountVectorizer.pkl"))
    print(f"--- Bag of words: CountVectorizer saved to Models/MRSA_CountVectorizer.pkl ---\n")
    print(f"BOW X shape : {X.shape}, Y shape: {y.shape}\n")

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    print(f"Train shapes X : {X_train.shape}, y : {y_train.shape}\n")
    print(f"Test shapes X : {X_test.shape}, y : {y_test.shape}\n")
    
   
    print("--- Training Naive Bayes Models ---")
    gnb = GaussianNB()
    mnb = MultinomialNB(alpha=1.0, fit_prior=True)
    bnb = BernoulliNB(alpha=1.0, fit_prior=True)
    
    gnb.fit(X_train, y_train)
    mnb.fit(X_train, y_train)
    bnb.fit(X_train, y_train) 

    joblib.dump(gnb, os.path.join(MODELS_DIR, "MRSA_gnb.pkl"))
    joblib.dump(mnb, os.path.join(MODELS_DIR, "MRSA_mnb.pkl"))
    joblib.dump(bnb, os.path.join(MODELS_DIR, "MRSA_bnb.pkl"))
    
    print(f"Trained models saved successfully to the '{MODELS_DIR}' directory.\n")
    
    
    ypg = gnb.predict(X_test)
    ypm = mnb.predict(X_test)
    ypb = bnb.predict(X_test)

    print("--- Model Performance on Test Set ---")
    print(f"Gaussian accuracy = {round(accuracy_score(y_test, ypg), 4) * 100} %")
    print(f"Multinomial accuracy = {round(accuracy_score(y_test, ypm), 4) * 100} %")
    print(f"Bernoulli accuracy = {round(accuracy_score(y_test, ypb), 4) * 100} %")
    print("-" * 50)
    
if __name__ == "__main__":
    run_sentiment_analysis()