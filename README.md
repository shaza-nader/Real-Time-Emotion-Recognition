# Real-Time Face Emotion Recognition System

# Project Description
An interactive, modular Streamlit web platform that interfaces with local or cloud hardware webcams to continuously monitor human faces, run real-time structural analysis frameworks, and predict current emotional expressions using deep convolutional structures.

## Used Dataset
Optimized using the **FER2013** (Facial Expression Recognition) dataset consisting of 48x48 grayscale pixel mappings across 7 distinct human emotion categories.

## Architecture and Structure
- `app.py`: Streamlit responsive web environment hosting interface blocks and capture hooks.
- `model/preprocess.py`: Input pipeline validations.
- `model/predict.py`: Inference execution framework leveraging deep neural network architectures.
- `requirements.txt`: Environment package manifest.

## How to Run Locally
1. Clone the repository directory structure to your environment.
2. Install all required software packages:
   ```bash
   pip install -r requirements.txt

## Execute the server thread via your terminal:
Bash
streamlit run app.py

## Team Contributions
Shaza: Built the core Streamlit reactive modular UI architecture (app.py), handled preprocessing validations (preprocess.py), and managed live capture frame conversions.
