import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Define the path to the organized dataset
# DATASET_PATH = "datasets/crema_d/organized"
DATASET_NAME = "ser_jeans_voice"
DATASET_PATH = f"datasets/{DATASET_NAME}/organized"
OUTPUT_FOLDER = f"evaluation_outputs/{DATASET_NAME}"

# Define the list of supported labels (CREMA-D labels)
CREMA_D_LABELS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad"]


# Load the pipeline
def load_pipeline(
    model="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8",
):
    # tried model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" really bad
    print("Loading the Wav2Vec2 pipeline...")
    print("model name:", model)
    pipe = pipeline("audio-classification", model=model)
    return pipe


# Preprocess the audio file
def preprocess_audio(file_path, target_sr=16000):
    # Load audio and resample to 16kHz
    signal, sr = librosa.load(file_path, sr=target_sr)
    return signal


# Perform prediction using the pipeline
def predict_emotion(pipe, signal):
    result = pipe(signal, sampling_rate=16000)
    # Extract the top prediction from the model
    predicted_label = result[0]["label"]
    return predicted_label


# Example usage

audio_path = (
    "/home/jeans/nvaitc/ser/datasets/crema_d/organized/Angry/1091_WSI_ANG_XX.wav"
)
pipe = load_pipeline()

signal = preprocess_audio(audio_path, target_sr=16000)
result = predict_emotion(pipe, signal)
print(result)
