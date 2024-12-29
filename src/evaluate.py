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
CREMA_D_LABELS = ["Happy", "Neutral", "Sad"]

# Define the mapping between CREMA-D labels and the model's RAVDESS labels
MODEL_LABELS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]
LABEL_MAPPING = {
    "Angry": "angry",
    "Disgust": "disgust",
    "Fearful": "fearful",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
}


# Load the pipeline
def load_pipeline(
    model="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8",
):
    # tried model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" really bad
    print("Loading the Wav2Vec2 pipeline...")
    print("model name:", model)
    pipe = pipeline("audio-classification", model=model)
    return pipe


# Load dataset files and labels
def load_dataset(dataset_path, crema_d_labels):
    print("Loading dataset...")
    files = []
    labels = []
    for label in crema_d_labels:
        folder = os.path.join(dataset_path, label)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    files.append(os.path.join(folder, file))
                    labels.append(label)
    return files, labels


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


# Map CREMA-D true labels to model-compatible labels
def map_true_labels_to_model_labels(true_labels):
    mapped_labels = []
    for true_label in true_labels:
        mapped_labels.append(LABEL_MAPPING[true_label])
    return mapped_labels


# Save the confusion matrix as an image
def save_confusion_matrix(cm, labels, output_folder):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_file = os.path.join(output_folder, "confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()  # Close the plot to avoid display issues
    print(f"Confusion matrix saved to {cm_file}")


# Custom function to calculate classification metrics
def calculate_metrics(true_labels, predicted_labels, labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    tp_per_class = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    total_predicted_per_class = cm.sum(axis=0)
    total_samples = cm.sum()

    # Precision, Recall, F1-Score per class
    precision = tp_per_class / total_predicted_per_class
    recall = tp_per_class / total_per_class
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Handle NaNs in precision and recall
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    # Overall accuracy
    accuracy = np.sum(tp_per_class) / total_samples

    # Print the classification report
    print("\nClassification Report:")
    print(f"{'Label':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    for i, label in enumerate(labels):
        print(f"{label:<12}{precision[i]:<12.4f}{recall[i]:<12.4f}{f1_score[i]:<12.4f}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    return accuracy, precision, recall, f1_score, cm


# Evaluate model predictions against ground truth labels
def evaluate_model(pipe, files, true_labels, output_folder):
    print("Evaluating model...")
    model_predictions = []
    for i, file in enumerate(files):
        # Preprocess the audio
        signal = preprocess_audio(file)
        # Predict emotion
        pred = predict_emotion(pipe, signal)
        model_predictions.append(pred)
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files...")

    # Map true labels to model-compatible labels
    mapped_true_labels = map_true_labels_to_model_labels(true_labels)

    # Calculate metrics
    accuracy, precision, recall, f1_score, cm = calculate_metrics(
        mapped_true_labels, model_predictions, MODEL_LABELS
    )

    # Save confusion matrix
    save_confusion_matrix(cm, MODEL_LABELS, output_folder)

    return model_predictions, accuracy, precision, recall, f1_score


# Ensure output folder exists
def prepare_output_folder(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)


# Main function to run inference and evaluation
if __name__ == "__main__":
    # Step 1: Ensure output folder exists
    prepare_output_folder(OUTPUT_FOLDER)

    # Step 2: Load the pipeline

    pipe = load_pipeline(
        model="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8"
    )

    # pipe = load_pipeline(
    #     model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    # )

    # Step 3: Load the dataset
    files, true_labels = load_dataset(DATASET_PATH, CREMA_D_LABELS)

    # Step 4: Evaluate the model and save outputs
    predictions, accuracy, precision, recall, f1_score = evaluate_model(
        pipe, files, true_labels, OUTPUT_FOLDER
    )
