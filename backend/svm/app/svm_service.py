from flask import Flask, request, jsonify
import base64
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Load the pre-trained SVM model
model = joblib.load("svm_model.pkl")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Decode WAV file
        data = request.json
        wav_music = base64.b64decode(data['wav_music'])
        
        # Extract features
        features = extract_features(wav_music)  # Implement feature extraction
        
        # Predict genre
        genre = model.predict([features])
        return jsonify({"genre": genre[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def extract_features(wav_data):
    # Dummy function - replace with real feature extraction
    return np.random.rand(10)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
