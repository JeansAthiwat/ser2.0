import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Define the list of supported emotion labels from this model dataset
# ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


# Function to load the emotion recognition model pipeline
def load_pipeline(
    model="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8",
):
    print("Loading the Wav2Vec2 pipeline...")
    print("Model name:", model)
    # Load the audio classification pipeline
    pipe = pipeline("audio-classification", model=model)
    return pipe


# Function to preprocess the audio file
def preprocess_audio(file_path, target_sr=16000):
    # Load the audio file and resample it to the target sampling rate
    signal, sr = librosa.load(file_path, sr=target_sr)
    return signal


# Function to predict emotion using the pipeline
def predict_emotion(pipe, signal):

    # Use the pipeline to make predictions
    result = pipe(signal, sampling_rate=16000)

    # result returns a list of dict of top 5 predictions
    # Extract the top predicted label from the results
    # Example : [{'score': 0.8223342895507812, 'label': 'angry'}, {'score': 0.06475415080785751, 'label': 'surprised'}, {'score': 0.053048718720674515, 'label': 'disgust'}, {'score': 0.03450959175825119, 'label': 'happy'}, {'score': 0.009342065081000328, 'label': 'fearful'}]
    # print(result)

    predicted_label = result[0][
        "label"
    ]  # left most is the top prediction return just the label
    return predicted_label


# Example usage of the code
if __name__ == "__main__":
    # Path to the audio file for emotion prediction
    audio_path = "datasets/example_angry_voice.wav"

    # Load the emotion recognition model pipeline
    pipe = load_pipeline()

    # Preprocess the audio file (convert it to a 16kHz signal MANDATORY)
    signal = preprocess_audio(audio_path, target_sr=16000)

    # Predict the emotion of the audio signal
    result = predict_emotion(pipe, signal)

    # Print the predicted emotion
    print(f"Predicted Emotion: {result}")
