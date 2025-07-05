# Generalized Deep Learning for Exoskeleton Control

This repository contains the code and documentation for my internship project at the Defence Research and Development Organisation (DRDO), focused on payload and intention classification for low-back exoskeletons.

## Project Overview

This project moves beyond traditional subject-specific models by developing a **robust, generalized deep learning framework**. The goal is to enable adaptive control for low-back exoskeletons without requiring time-consuming, individual user calibration, making the technology more practical for real-world industrial deployment.

The system uses data from wearable Inertial Measurement Units (IMUs) to classify both the user's activity (intention) and the weight of a lifted object (payload).

### Key Achievements
- **Generalized Payload Model (BiLSTM):** Achieved **93% accuracy** in classifying payloads of 0-15 kg.
- **Generalized Intention Model (CNN-BiLSTM):** Dramatically improved accuracy from a **42% baseline to 88%** by using SMOTE for class imbalance and a hybrid architecture for feature extraction.
- **Outperformed Baseline:** The generalized payload model's accuracy surpassed the 87.14% median accuracy of the original subject-specific models.

## Repository Structure
- `IMU4 (2).ipynb`: The main Google Colab notebook containing data preprocessing, model training, evaluation, and the Gradio demo setup.
- `README.md`: This project overview.

## Methodology
1.  **Data Source:** Publicly available dataset from Zenodo, containing IMU recordings from 12 participants.
2.  **Data Preprocessing:** Aggregated data from all users, applied a sliding window technique, and normalized features using `StandardScaler`.
3.  **Class Imbalance:** Utilized the **Synthetic Minority Over-sampling Technique (SMOTE)** on the training data to create a more balanced dataset.
4.  **Model Architectures:**
    - **Intention Classification:** A hybrid 1D Convolutional Neural Network (CNN) combined with a Bidirectional LSTM (BiLSTM).
    - **Payload Classification:** A Bidirectional LSTM (BiLSTM).
5.  **Tools & Frameworks:** Python, TensorFlow/Keras, Scikit-learn, Pandas, imblearn, and Gradio.

## How to Run
The primary code is contained within the Colab notebook (`IMU4 (2).ipynb`). To run it, you will need to upload it to Google Colab, mount your Google Drive, and ensure the necessary dataset files (as described in the notebook) are present in the specified Drive directory.
