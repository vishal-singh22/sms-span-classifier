import streamlit as st
import pickle
import os
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import traceback

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# 📌 Function to preprocess text
def transform_text(text):
    """Converts text to lowercase, removes stopwords, punctuations, and applies stemming."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# 📌 Function to train and save the model
def train_and_save_model(vectorizer_path, model_path):
    """Trains a new model and saves it if no pre-trained model exists."""
    try:
        # Load dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        df['text'] = df['text'].apply(transform_text)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)  # **Ensure model is trained**

        # Save vectorizer & model
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf, f)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print("✅ Model trained and saved successfully!")
        return tfidf, model

    except Exception as e:
        print(f"❌ Error training model: {e}")
        return None, None

# 📌 Function to load or train the model
def load_or_train_model():
    """Loads existing model or trains a new one if needed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
    model_path = os.path.join(current_dir, 'model.pkl')

    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        try:
            with open(vectorizer_path, 'rb') as f:
                tfidf = pickle.load(f)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Ensure model is trained before returning
            check_is_fitted(model)
            print("✅ Model loaded successfully!")
            return tfidf, model

        except Exception as e:
            print(f"⚠️ Model loading failed: {e}. Retraining...")

    # Train & Save Model if Loading Fails
    return train_and_save_model(vectorizer_path, model_path)

# Load or train model at startup
tfidf, model = load_or_train_model()

# 📌 Streamlit App UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("✍️ Enter your message below:")

if st.button('🔍 Predict'):
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error(traceback.format_exc())
