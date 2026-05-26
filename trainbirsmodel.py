import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = r"C:\Users\Admin\soundsfiles\bird_dataset"

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    feature = np.mean(mfccs.T, axis=0)

    return feature
    print(os.listdir(DATASET_PATH))


for bird_name in os.listdir(DATASET_PATH):

    bird_folder = os.path.join(DATASET_PATH, bird_name)

    if os.path.isdir(bird_folder):

        for file in os.listdir(bird_folder):

            file_path = os.path.join(bird_folder, file)

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(bird_name)

            except Exception as e:
                print("Error:", file_path)
                
X = np.array(X)
y = np.array(y)
print("bird name",y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

joblib.dump(model, "bird_sound_model.pkl")

print("Model Saved")

