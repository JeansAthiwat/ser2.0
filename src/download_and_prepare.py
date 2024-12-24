import os
import zipfile
import shutil
from pathlib import Path
import requests

# Step 1: Define constants and label mapping
LABEL_MAPPING = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fearful",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

CREMA_D_URL = (
    "your_crema_d_dataset_url_here"  # Provide the download URL for CREMA-D dataset
)

# Define folder structure
BASE_FOLDER = Path("datasets/crema_d")
RAW_FOLDER = BASE_FOLDER / "raw"  # For storing raw ZIP files
EXTRACTED_FOLDER = BASE_FOLDER / "extracted"  # For storing extracted contents
ORGANIZED_FOLDER = BASE_FOLDER / "organized"  # For storing organized datasets


# Step 2: Function to download the ZIP file
def download_dataset(url, output_path):
    if not output_path.exists():
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Downloaded dataset to {output_path}.")
    else:
        print(f"Dataset already exists at {output_path}.")


# Step 3: Function to extract ZIP file
def extract_zip(zip_path, extract_to, overwrite=True):
    if (not extract_to.exists()) or overwrite:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}.")
    else:
        print(f"Dataset already extracted to {extract_to}.")


# Step 4: Function to organize files into subfolders based on labels
def organize_crema_d(dataset_folder, output_folder, label_mapping):
    print(f"Organizing dataset into subfolders at {output_folder}...")
    audio_folder = dataset_folder / "AudioWAV"
    if not audio_folder.exists():
        raise FileNotFoundError(f"AudioWAV folder not found in {dataset_folder}.")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for file in audio_folder.iterdir():
        if file.suffix == ".wav":
            # Extract label from filename
            label_key = file.stem.split("_")[2]
            if label_key in label_mapping:
                label = label_mapping[label_key]
                label_folder = output_folder / label
                os.makedirs(label_folder, exist_ok=True)
                shutil.copy(file, label_folder / file.name)
            else:
                print(f"Skipping file {file.name} (label {label_key} not mapped).")
    print(f"Dataset organized at {output_folder}.")


# Step 5: Main function to handle the entire process
def prepare_crema_d_dataset(
    zip_url, raw_folder, extracted_folder, organized_folder, label_mapping
):
    # Ensure base folders exist
    raw_folder.mkdir(parents=True, exist_ok=True)
    extracted_folder.mkdir(parents=True, exist_ok=True)
    organized_folder.mkdir(parents=True, exist_ok=True)

    # Step 5.1: Download dataset
    zip_path = raw_folder / "CREMA-D.zip"
    download_dataset(zip_url, zip_path)

    # Step 5.2: Extract dataset
    extract_zip(zip_path, extracted_folder)

    # Step 5.3: Organize dataset
    organize_crema_d(extracted_folder, organized_folder, label_mapping)


# Run the preparation pipeline for CREMA-D
if __name__ == "__main__":
    prepare_crema_d_dataset(
        zip_url=CREMA_D_URL,
        raw_folder=RAW_FOLDER,
        extracted_folder=EXTRACTED_FOLDER,
        organized_folder=ORGANIZED_FOLDER,
        label_mapping=LABEL_MAPPING,
    )
