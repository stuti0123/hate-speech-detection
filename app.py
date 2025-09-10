import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

@st.cache_resource
def load_model():
    model = pickle.load(open("multi_hate_model.pkl", "rb"))
    vectorizer = pickle.load(open("multi_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
colors = {"Hate Speech": "red", "Offensive Language": "orange", "Neither": "green"}


# Streamlit UI
st.set_page_config(page_title="Hate Speech Classifier", page_icon="🚨", layout="wide")

st.title("🚨 Multi-class Hate/Offensive Language Detector")
st.markdown(
    """
    This app uses a **Machine Learning model** trained on a labeled dataset to detect:  
    - 🟥 Hate Speech  
    - 🟧 Offensive Language  
    - 🟩 Neither  

    Upload text below to classify 👇
    """
)


# User Input
text = st.text_area("✍️ Enter text to classify:")

if st.button("Classify"):
    if text.strip():
        cleaned = text.lower()
        transformed = vectorizer.transform([cleaned])
        pred = model.predict(transformed)[0]
        label = label_map[pred]

        st.subheader("📊 Prediction Result")
        st.markdown(f"**Prediction:** <span style='color:{colors[label]}; font-size:20px'>{label}</span>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter some text.")


# Dataset Insights
st.sidebar.title("📊 Dataset Explorer")

if st.sidebar.checkbox("Show dataset preview"):
    df = pd.read_csv("labeled_data.csv")
    st.dataframe(df.head())

if st.sidebar.checkbox("Class distribution"):
    df = pd.read_csv("labeled_data.csv")
    dist = df['class'].value_counts().map(label_map)
    fig, ax = plt.subplots()
    sns.countplot(x="class", data=df, ax=ax, palette="Set2")
    ax.set_xticklabels([label_map[i] for i in ax.get_xticks()])
    ax.set_title("Class Distribution")
    st.pyplot(fig)

if st.sidebar.checkbox("Word Cloud of Offensive Texts"):
    df = pd.read_csv("labeled_data.csv")
    text_data = " ".join(df[df['class'] == 1]['tweet'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text_data)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

