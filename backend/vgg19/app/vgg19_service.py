from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained VGG19 model
model = load_model("vgg19_model.h5")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Decode WAV file
        data = request.json
        wav_music = base64.b64decode(data['wav_music'])
        
        # Extract features
        features = extract_features(wav_music)  # Implement feature extraction
        
        # Predict genre
        predictions = model.predict(np.expand_dims(features, axis=0))
        genre = np.argmax(predictions)
        return jsonify({"genre": int(genre)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def extract_features(wav_data):
    # Dummy function - replace with real feature extraction
    return np.random.rand(128, 128, 3)  # Example shape for VGG19 input

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
