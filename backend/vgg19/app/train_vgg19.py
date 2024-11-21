import os
import numpy as np
import librosa
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub  # Library for Kaggle datasets

# Step 1: Download dataset from Kaggle
print("Downloading dataset...")
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print("Path to dataset files:", path)

# Set the path to the dataset (update based on actual downloaded structure)
data_folder = os.path.join(path, "genres_original")  # Adjust based on dataset structure

# Step 2: Extract Mel Spectrogram Features
def extract_mel_spectrogram(file_path):
    signal, rate = librosa.load(file_path, sr=None)  # Load file without resampling
    S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=2048, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB[:128, :128]  # Resize to 128x128

def load_data(folder):
    labels = []
    features = []

    if not os.path.exists(folder):
        print(f"Dataset folder not found: {folder}")
        return features, labels

    # Loop through each genre folder and extract features
    for genre in os.listdir(folder):
        genre_folder = os.path.join(folder, genre)
        if os.path.isdir(genre_folder):
            for file in os.listdir(genre_folder):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_folder, file)
                    features.append(extract_mel_spectrogram(file_path))  # Extract features
                    labels.append(genre)  # Append the genre label

    # Check if labels and features are populated
    if len(labels) == 0:
        print(f"No labels found in folder: {folder}")
    else:
        print(f"Labels found: {len(labels)}")

    X = np.array(features).reshape(-1, 128, 128, 1)  # Reshape for CNN
    y = np.array(labels)

    return X, y

# Step 3: Load and preprocess the data
print("Loading and preprocessing data...")
data_folder = "././data_folder"
X, y = load_data(data_folder)

# Check if the labels are correctly loaded
print(f"Loaded labels: {y[:10]}")  # Print the first 10 labels for verification

# Ensure that labels are not empty
if len(y) == 0:
    raise ValueError("No data found. Check the dataset extraction or path.")

# Encode labels as integers and then as one-hot vectors
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers
print(f"Encoded labels: {y_encoded[:10]}")  # Print first 10 encoded labels

# Ensure that y_encoded is not empty
if len(y_encoded) == 0:
    raise ValueError("Encoded labels are empty. Check the label encoding process.")

# One-hot encode labels
y_categorical = to_categorical(y_encoded)


# Step 3: Load and preprocess the data
print("Loading and preprocessing data...")
X, y = load_data(data_folder)

# Encode labels as integers and then as one-hot vectors
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers
y_categorical = to_categorical(y_encoded)   # One-hot encode labels

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Step 4: Define the VGG19 Model
def create_vgg19_model(input_shape, num_classes):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False  

    model = Sequential([
        base_model,
        Flatten(),  # Flatten the output of VGG19
        Dense(256, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    return model

# Create and compile the VGG19 model
print("Creating the VGG19 model...")
input_shape = (128, 128, 1)  # Spectrogram input shape (128x128)
num_classes = y_categorical.shape[1]  # Number of classes in the dataset
model = create_vgg19_model(input_shape, num_classes)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
print("Training the model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Step 6: Save the trained model
model.save("vgg19_model.h5")
print("Model training completed and saved as 'vgg19_model.h5'")
