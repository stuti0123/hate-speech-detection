# 🛡️ Hate Speech Detection Web App  

A simple **Machine Learning + Streamlit** project that detects **Hate Speech**, **Offensive Language**, and **Neither** from user input text.  
This project demonstrates text preprocessing, feature extraction, model training, and deployment using Streamlit Cloud.  

---

## 🚀 Features  
- ✅ Data Preprocessing (cleaning text, removing stopwords, punctuation, and URLs)  
- ✅ Feature Extraction with **CountVectorizer** and **TF-IDF**  
- ✅ Logistic Regression model for classification  
- ✅ Evaluation using accuracy, confusion matrix, and classification report  
- ✅ Interactive **Streamlit Web App** for predictions  

---

## 📂 Project Structure  

hate-speech-detection/
│-- app.py # Streamlit web app
│-- hate_speech_detection.ipynb # Colab notebook (training & analysis)
│-- requirements.txt # Dependencies
│-- README.md # Project documentation

yaml
Copy code

---

## 🖥️ How to Run Locally  

1. Clone this repository:  
   ```bash
   git clone https://github.com/stuti0123/hate-speech-detection.git
   cd hate-speech-detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the Streamlit app:
   ```bash
   streamlit run app.py

🌐 Deployment
This app can be deployed easily on Streamlit Cloud:

Upload your project repo to GitHub

Sign in to Streamlit Cloud and connect your repo

Deploy the app with one click

🔗 Live Demo Link: (Add once deployed)

📊 Dataset
Dataset used: Hate Speech and Offensive Language Dataset

7 columns (including tweet, class, etc.)

3 Labels:

0 → Hate Speech

1 → Offensive Language

2 → Neither

📈 Model Performance
Accuracy: ~90% with CountVectorizer + Logistic Regression

TF-IDF also tested, with slightly different performance

Confusion matrix and classification reports included

✨ Future Improvements
Use Deep Learning models (LSTMs, BERT) for better accuracy

Add data visualizations of predictions in the web app

Enhance UI/UX with advanced Streamlit components

Integrate with APIs for real-time hate speech monitoring

🤝 Contributing
Contributions are welcome! Feel free to open issues and submit pull requests.

📜 License
This project is licensed under the MIT License.