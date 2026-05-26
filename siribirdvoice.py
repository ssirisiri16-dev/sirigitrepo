import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

st.set_page_config(page_title="Bird Sound Detector", layout="wide")

st.title("🐦 Bird Sound Detection App")
st.markdown("""
<h5 class="title">
🐦 Bird Sound Detection App
</h5>

<style>

.title {
    text-align: center;
    font-size: 55px;
    color: white;
    font-weight: bold;

    padding: 20px;

    border-radius: 20px;

    background: linear-gradient(to right, #ff512f, #dd2476);

    box-shadow: 0px 0px 30px rgba(255,255,255,0.8);

    animation: pop 2s infinite;
}

/* Pop Animation */
@keyframes pop {

    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.08);
    }

    100% {
        transform: scale(1);
    }
}
/* Main App Background */
.stApp {
    background-color: pink;
}


/* Browse Files Button */
.stFileUploader button {
    background-color:maroon;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}

/* Hover Effect */
.stFileUploader button:hover {
    background-color: darkred;
    color: yellow;
}


</style>

<div class="bird">🕊️</div>

""", unsafe_allow_html=True)


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