import streamlit as st
import pickle

model = pickle.load(open("multi_hate_model.pkl", "rb"))
vectorizer = pickle.load(open("multi_vectorizer.pkl", "rb"))

st.title("🚨 Multi-class Hate/Offensive Language Detector")
text = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if text.strip():
        cleaned = text.lower()
        transformed = vectorizer.transform([cleaned])
        pred = model.predict(transformed)[0]
        label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
        st.subheader(f"Prediction: {label_map[pred]}")
    else:
        st.warning("Please enter some text.")
