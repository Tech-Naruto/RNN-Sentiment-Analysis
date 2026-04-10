import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.datasets import imdb
import streamlit as st


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}


# Load the pre-trained model
model = load_model('simple_rnn_imdb.keras')


# Helper Function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, -1) + 3 for word in words]
    encoded_review = [1] + encoded_review
    padded_review = pad_sequences([encoded_review], maxlen=300, padding='pre')
    return padded_review


# Prediction function
def predict_sentiment(review):
    preprocessed_review = preprocess_review(review)

    prediction = model.predict(preprocessed_review)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0] 


# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

review = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment, confidence_score = predict_sentiment(review)
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: {confidence_score:.4f}")
    else:
        st.write("Please enter a movie review to predict its sentiment.")