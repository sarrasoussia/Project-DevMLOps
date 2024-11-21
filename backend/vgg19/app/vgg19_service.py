from flask import Flask, request, jsonify
import pickle
import base64
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load pre-trained VGG19 model
vgg19_model = load_model("vgg19_model.h5")

# Function to extract Mel Spectrogram features
def extract_features(signal, rate):
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    # Reshape for VGG19 input (128x128x1 required for a spectrogram image)
    return np.expand_dims(S_DB[:128, :128], axis=-1)

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
        features = np.expand_dims(features, axis=0)  # Add batch dimension

        # Predict genre
        prediction = vgg19_model.predict(features)
        genre_index = np.argmax(prediction, axis=1)[0]
        genres = {0: "normal", 1: "abnormal"}  # Map prediction to genre
        return jsonify({"genre": genres[genre_index]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
