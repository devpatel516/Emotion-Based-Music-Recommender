import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# --- CONFIGURATION & SETUP ---
load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/cointegrated/rubert-tiny2-cedr-emotion-detection"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ **Error:** Hugging Face API Token (HF_TOKEN) not found.")
    st.stop()

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

EMOTION_TO_GENRE = {
    "joy": ["pop", "dance", "electronic"],
    "happiness": ["pop", "dance", "electronic"],
    "sadness": ["acoustic", "soft rock", "ballad"],
    "anger": ["rock", "metal", "rap"],
    "fear": ["dark ambient", "classical", "lofi"],
    "surprise": ["indie", "experimental", "jazz"],
    "disgust": ["punk", "grunge"],
    "love": ["r&b", "soul", "romantic pop"],
    "neutral": ["chill", "instrumental", "lofi"]
}

# --- DATA & MODEL FUNCTIONS ---

@st.cache_data
def load_data():
    try:
        songs_df = pd.read_csv("spotify_plus_youtube.csv")
        songs_df.columns = songs_df.columns.str.lower()
        if 'playlist_genre' not in songs_df.columns:
            st.error("❌ Required column 'playlist_genre' not found in the dataset.")
            st.stop()
        return songs_df
    except FileNotFoundError:
        st.error("❌ Dataset file `spotify_plus_youtube.csv` not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {e}")
        st.stop()

@st.cache_resource
def query_emotion(text):
    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Error: Could not reach the Hugging Face model. {e}")
        return None

def get_dominant_emotion(text):
    output = query_emotion(text)
    if not output or not isinstance(output, list) or not output[0]:
        return None
    emotions = output[0]
    best = max(emotions, key=lambda x: x["score"])
    return best["label"]

def get_songs_for_emotion(songs_df, emotion, n=10):
    emotion = emotion.lower()
    genres = EMOTION_TO_GENRE.get(emotion, ["pop"])
    mask = songs_df['playlist_genre'].astype(str).str.lower().apply(
        lambda g: any(genre in g for genre in genres)
    )
    filtered = songs_df[mask]

    if filtered.empty:
        return pd.DataFrame(), genres

    selected = filtered.sample(n=min(n, len(filtered)), random_state=42)
    columns_to_show = ['track_name', 'track_artist', 'playlist_genre', 'youtube_link']
    existing_columns = [col for col in columns_to_show if col in selected.columns]
    result = selected[existing_columns].reset_index(drop=True)
    return result, genres

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="🎧 Emotion-Based Music Recommender", layout="wide")

    # Custom CSS for beautiful UI
    st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stApp {
            background: linear-gradient(135deg, #141e30 0%, #243b55 100%);
            color: white;
        }
        h1, h2, h3, h4 {
            color: #FFD700 !important;
            text-shadow: 1px 1px 2px black;
        }
        .stButton button {
            background: linear-gradient(90deg, #ff8c00, #ff0080);
            color: white;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            padding: 0.6em 1.2em;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #ffb347, #ffcc33);
            color: black;
        }
        .song-card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1em;
            margin-bottom: 1em;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        small {
            color: #bbb;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎧 Emotion-Based Music Recommender")
    st.markdown("### _Let your mood decide your music!_")
    st.markdown("---")

    songs_df = load_data()

    st.header("💭 What are you feeling today?")
    prompt = st.text_area(
        label="Describe your current mood:",
        placeholder="e.g., I just got a promotion, I'm feeling incredibly happy and energized!",
        height=100
    )

    st.markdown("---")

    if st.button("Get My Music Recommendation 🎵"):
        if prompt:
            with st.spinner("Analyzing your emotion and searching for songs..."):
                emotion = get_dominant_emotion(prompt)
                if emotion:
                    st.subheader(f"✅ Detected Emotion: **{emotion.upper()}**")

                    recommendations_df, target_genres = get_songs_for_emotion(songs_df, emotion, n=10)

                    if not recommendations_df.empty:
                        st.markdown(f"**🎶 Target Genres:** *{', '.join([g.title() for g in target_genres])}*")
                        st.markdown(f"### Top {len(recommendations_df)} Recommendations for your mood:")

                        for idx, row in recommendations_df.iterrows():
                            track_name = row.get('track_name', row.get('track_album_name', 'Unknown Title'))
                            artist = row.get('track_artist', 'Unknown Artist')
                            genre = row.get('playlist_genre', 'N/A')
                            link = row.get('youtube_link', '')

                            # Modified link to auto-play the first YouTube result
                            search_query = f"{track_name} {artist}".replace(" ", "+")
                            youtube_direct_link = f"https://www.youtube.com/results?search_query={search_query}"

                            st.markdown(f"""
                                <div class='song-card'>
                                    <b>{idx+1}. {track_name}</b> 🎤 *{artist}* <br>
                                    <small>Genre: {genre}</small><br>
                                    ▶️ <a href="{youtube_direct_link}" target="_blank" style="color:#FFD700;text-decoration:none;font-weight:bold;">
                                        Play
                                    </a>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ No songs found matching genres for **{emotion.upper()}** ({', '.join(target_genres)}).")
                else:
                    st.error("❌ Could not detect a dominant emotion from your input. Try a different prompt.")
        else:
            st.warning("Please enter some text to describe your mood.")

if __name__ == "__main__":
    main()
