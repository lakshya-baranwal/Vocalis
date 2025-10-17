# Early Detection of Parkinson’s Disease and Vocal Cord Paralysis
### Developed in BitNBuild’25 Finals

## Overview
This project addresses the challenge of early detection of Parkinson’s Disease and Vocal Cord Paralysis through voice analysis.
Speech changes are among the earliest symptoms of both disorders, and our system leverages deep learning on raw voice data to automatically identify these subtle cues.

## Approach
We developed a Hybrid CRNN (Convolutional Recurrent Neural Network) model that combines:

- **2D Convolutional Neural Network (CNN)** for spatial (acoustic feature) extraction  
- **Bidirectional LSTM (RNN)** for temporal (speech pattern) modeling  

### Model Pipeline
1. **Input:** Raw audio file (.wav)  
2. **Preprocessing:** Conversion to **Mel Spectrogram** — a visual representation of sound capturing both frequency and time variations  
3. **Model Inference:** The hybrid CRNN processes the spectrogram to classify potential early indicators  

## Model Architecture

### CNN Component
Extracts static acoustic features such as:
- Hoarseness — irregular or noisy spectrogram textures  
- Vocal tremor — wavy or shaky frequency patterns  
- Monotonous pitch — flat or uniform spectral bands  

### RNN (Bi-LSTM) Component
Captures dynamic speech features including:
- Rhythm and cadence  
- Speech rate  
- Temporal consistency and variations  

## Why Spectrograms?
Unlike traditional ML models dependent on hand-engineered features (e.g., MFCCs, jitter, shimmer), our approach uses **spectrogram-based inputs** that preserve the richness of raw speech signals, enabling detection of subtle early-stage anomalies in vocal performance.

## Tech Stack
- **Languages:** Python  
- **Frameworks:** PyTorch / TensorFlow  
- **Audio Processing:** Librosa  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / Colab  

