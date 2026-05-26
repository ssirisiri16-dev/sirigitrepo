import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

st.set_page_config(page_title="Bird Sound Detector", layout="wide")

st.title("🐦 Bird Sound Detection App")

model = joblib.load("bird_sound_model.pkl")

def extract_features(file_path):

    audio, sr = librosa.load(file_path, duration=5)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    feature = np.mean(mfccs.T, axis=0)

    return feature

uploaded_file = st.file_uploader(
    "Upload Bird Audio",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    features = extract_features(temp_path)

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    st.success(f"Detected Bird: {prediction}")