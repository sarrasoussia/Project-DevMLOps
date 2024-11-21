from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np 
import librosa
import os

app = Flask(__name__)

model_path = "./models/genre_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# @app.route('/', methods=['GET'])
# def get():
#    return("hello svm_model")

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    # Save the file temporarily
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Process the audio file and extract features
    features = extract_features(file_path)
    os.remove(file_path)  # Remove the temp file after processing

    # Make prediction using the model
    prediction = model.predict([features])
    genre = prediction[0]

    return jsonify({'genre': genre})

def extract_features(file_path):
    """Extracts features from the audio file using Librosa."""
    y, sr = librosa.load(file_path, mono=True)
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return features


if __name__ == '__main__':
    os.makedirs("temp", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)