import pandas as pd
import random
from model import get_dominant_emotion

songs_df = pd.read_csv("spotify_plus_youtube.csv")

songs_df.columns = songs_df.columns.str.lower()

#print("Columns available:", songs_df.columns.tolist())

emotion_to_genre = {
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
text=input("Enter your prompt: ")
emotion = get_dominant_emotion(text)
print("\nDetected Emotion →", emotion.upper())

def get_songs_for_emotion(emotion, n=10):
    emotion = emotion.lower()
    genres = emotion_to_genre.get(emotion, ["pop"])

    if 'playlist_genre' not in songs_df.columns:
        raise ValueError("❌ 'genre' column not found in the dataset.")
    
    mask = songs_df['playlist_genre'].astype(str).str.lower().apply(
        lambda g: any(genre in g for genre in genres)
    )
    filtered = songs_df[mask]

    if filtered.empty:
        print(f"\n⚠️ No songs found for emotion '{emotion}' → genres {genres}")
        return pd.DataFrame()

    selected = filtered.sample(n=min(n, len(filtered)), random_state=42)
    columns_to_show = [col for col in ['track_album_name', 'track_artist', 'playlist_genre', 'youtube_link'] if col in filtered.columns]
    result = selected[columns_to_show].reset_index(drop=True)

    print(f"\n🎵 Top {len(result)} Recommended Songs for '{emotion.title()}' emotion:\n" + "-"*65)
    for idx, row in result.iterrows():
        print(f"{idx+1}. {row.get('track_album_name', 'Unknown Title')}  🎤 {row.get('track_artist', 'Unknown Artist')}")
        print(f"   Genre: {row.get('playlist_genre', 'N/A')}")
        print(f"   🔗 Link: {row.get('youtube_link', 'No link available')}\n")
    print("-"*65)
    
    return result

songs = get_songs_for_emotion(emotion)
