# 🔧 STEP 1: Install everything needed
!pip install streamlit deepface streamlit-webrtc scikit-learn -q
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!mv cloudflared-linux-amd64 cloudflared
!chmod +x cloudflared
!mv cloudflared /usr/local/bin

# 📁 STEP 2: Write all Python files needed
# face_mood_live.py
with open("face_mood_live.py", "w") as f:
    f.write('''
import av
from streamlit_webrtc import VideoTransformerBase
from deepface import DeepFace
import cv2

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = "Unknown"

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            self.last_emotion = emotion
            cv2.putText(img, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        except:
            cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img
''')

# text_mood.py
with open("text_mood.py", "w") as f:
    f.write('''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_text_model():
    texts = ["I am happy", "This is terrible", "I feel great", "I am sad", "I love this", "I hate this"]
    labels = ["happy", "sad", "happy", "sad", "happy", "sad"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    joblib.dump(model, "text_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if not os.path.exists("text_model.pkl"):
    train_text_model()

def predict_text_emotion(text):
    model = joblib.load("text_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    X = vectorizer.transform([text])
    return model.predict(X)[0]
''')

# recommender.py
with open("recommender.py", "w") as f:
    f.write('''
def get_recommendation(emotion):
    recs = {
        "happy": "🎵 Try 'Pharrell - Happy'",
        "sad": "🎵 Try 'Fix You - Coldplay'",
        "angry": "🎵 Try 'Weightless - Marconi Union'",
        "surprise": "🎵 Try 'Bohemian Rhapsody'",
        "fear": "🎵 Try 'Brave - Sara Bareilles'",
        "neutral": "🎵 Try 'Lofi Jazz Beats'"
    }
    return recs.get(emotion, "🎵 Explore more moods!")
''')

# app.py
with open("app.py", "w") as f:
    f.write('''
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from face_mood_live import EmotionDetector
from text_mood import predict_text_emotion
from recommender import get_recommendation

st.set_page_config(page_title="Moodify X Live", page_icon="🎧")
st.title("🎧 Moodify X: Real-time Mood Detection")

st.markdown("#### 🎥 Face-Based Emotion Detection (Live Webcam)")
ctx = webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)

if ctx.video_transformer:
    emotion = ctx.video_transformer.last_emotion
    if emotion != "Unknown":
        st.success(f"Detected Face Emotion: **{emotion}**")
        st.info(get_recommendation(emotion))

st.markdown("---")
st.header("🧠 Text-Based Emotion Detection")
user_text = st.text_input("How are you feeling right now?")
if user_text:
    text_mood = predict_text_emotion(user_text)
    st.success(f"Detected Text Emotion: **{text_mood}**")
    st.info(get_recommendation(text_mood))
''')

# 🚀 STEP 3: Run Streamlit & Expose with Cloudflared
import os
os.system("streamlit run app.py &")
!cloudflared tunnel --url http://localhost:8501 --no-autoupdate
