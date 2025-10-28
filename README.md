# 🎧 Emotion-Based Music Recommender

An **AI-powered Streamlit web application** that recommends songs based on a user’s emotional state using **NLP emotion detection** and **Spotify–YouTube metadata**.  
This project leverages **Hugging Face emotion analysis**, intelligent genre mapping, and a visually appealing Streamlit interface to personalize music recommendations.

---

## 🚀 Features

- 🧠 **Emotion Detection:** Uses Hugging Face model `rubert-tiny2-cedr-emotion-detection` to analyze user input text.  
- 🎵 **Smart Recommendations:** Maps emotions like joy, sadness, anger, and love to suitable music genres.  
- 🎧 **Spotify + YouTube Dataset:** Suggests real-world tracks with direct YouTube play links.  
- 💬 **Interactive Streamlit UI:** Simple and responsive interface with a custom CSS theme.  
- ⚡ **Fast and Optimized:** Uses Streamlit caching for efficient data and API handling.  

---

## 🧠 Tech Stack
| Category | Technologies |
|-----------|--------------|
| **Frontend/UI** | Streamlit, HTML/CSS |
| **Backend** | Python, Hugging Face Inference API |
| **Data Handling** | Pandas |
| **Dataset** | Spotify + YouTube Metadata |
| **Environment Management** | python-dotenv |
| **APIs** | Hugging Face API |
---

## 📦 Project Structure

Emotion-Based-Music-Recommender/
│
├── app.py 
├── extract.py 
├── model.py 
├── spotify_plus_youtube.csv 
├── requirements.txt 
├── .env 
└── README.md 
---
Add your huggingface api token in .env file

## Installation & Setup
pip install -r requirements.txt
streamlit run app.py
