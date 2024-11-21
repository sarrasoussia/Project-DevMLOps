import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.image import resize_with_crop_or_pad
import os

# Get absolute path for the saved model based on the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vgg19_model")

# Load the saved model
@st.cache_resource()
def load_saved_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model path {MODEL_PATH} does not exist.")
        return None
    try:
        model = load_model(MODEL_PATH)
        model.summary()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to build VGG19 model (for reference only, not used in this app)
def build_vgg_model(input_shape):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Load and preprocess audio file
def load_and_preprocess_file(file_path, target_shape=(130, 13)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=target_shape[0], fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        mel_spectrogram = resize_with_crop_or_pad(mel_spectrogram, target_shape[0], target_shape[1])
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(X_test, model):
    if model is not None:
        try:
            y_pred = model.predict(X_test)
            predicted_categories = np.argmax(y_pred, axis=1)
            unique_elements, counts = np.unique(predicted_categories, return_counts=True)
            max_count = np.max(counts)
            max_elements = unique_elements[counts == max_count]
            return max_elements[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    else:
        st.error("Model could not be loaded. Please check the model path or file.")
        return None

st.header("Music Genre Classification with VGG")

# Upload audio file
test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

# Save the file locally if uploaded
if test_mp3 is not None:
    BASE_DIR = os.getcwd()  # Or set to a specific path as needed

    # Set the file path for the archive.zip file
    filepath = os.path.join(BASE_DIR, "data_folder", "archive.zip")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Example of writing to a file (ensure test_mp3 is defined)
    with open(filepath, "wb") as f:
        f.write(test_mp3.getbuffer()) 

# Play audio button
if st.button("Play Audio") and test_mp3 is not None:
    st.audio(filepath)

# Predict button
if st.button("Predict") and test_mp3 is not None:
    with st.spinner("Please wait ..."):
        X_test = load_and_preprocess_file(filepath)
        model = load_saved_model()
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.markdown("**:blue[Model prediction]: It's a :red[{}] music**".format(label[result_index]))
        else:
            st.error("Prediction failed due to an error with the model.")
