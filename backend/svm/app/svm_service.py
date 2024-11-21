from flask import Flask, request, jsonify
import pickle
import librosa
import numpy as np
from sklearn import svm

app = Flask(__name__)

# Load pre-trained SVM model and feature processing configuration
svm_model = pickle.load(open("svm_model.pkl", "rb"))

# Function to extract Mel Spectrogram features
def extract_features(signal, rate):
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB.flatten()[:1200]

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Decode the WAV file from base64
        data = request.json
        wav_data = base64.b64decode(data["wav_music"])

        # Save WAV data to a temporary file
        with open("temp.wav", "wb") as f:
            f.write(wav_data)

        # Load the WAV file
        signal, rate = librosa.load("temp.wav")

        # Extract features
        features = extract_features(signal, rate)

        # Predict genre
        prediction = svm_model.predict([features])[0]
        genres = {0: "normal", 1: "abnormal"}  # Map prediction to genre
        return jsonify({"genre": genres[prediction]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
