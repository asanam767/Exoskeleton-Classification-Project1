import gradio as gr
import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model

# Configuration
project_dir = os.path.dirname(os.path.abspath(__file__))  # Use current directory for deployment
payload_model_path = os.path.join(project_dir, "payload_model_improved_combined.keras")
intention_model_path = os.path.join(project_dir, "intention_model_improved_combined.keras")
scaler_payload_path = os.path.join(project_dir, "scaler_payload_combined.pkl")
scaler_intention_path = os.path.join(project_dir, "scaler_intention_combined.pkl")
window_size = 100
num_features = 30

# Load Models and Scalers
print("Loading models and scalers...")
try:
    payload_model = load_model(payload_model_path)
    intention_model = load_model(intention_model_path)
    with open(scaler_payload_path, 'rb') as f:
        scaler_payload = pickle.load(f)
    with open(scaler_intention_path, 'rb') as f:
        scaler_intention = pickle.load(f)
    print("Models and scalers loaded successfully.")
except Exception as e:
    print(f"ERROR loading models or scalers: {e}")
    raise

# Prediction Function
def predict_from_upload(file, model_type, start_row):
    try:
        start_row = int(start_row)
        df = pd.read_csv(file)
        if start_row < 0 or start_row > len(df) - window_size:
            return {"error": f"Start row must be between 0 and {len(df) - window_size}"}
        if 'target' not in df.columns:
            return {"error": "'target' column not found in CSV"}
        feature_columns = df.columns.drop('target')
        if len(feature_columns) != num_features:
            return {"error": f"Expected {num_features} features, found {len(feature_columns)}"}

        segment_data = df.iloc[start_row : start_row + window_size][feature_columns].values
        if model_type == "Payload":
            scaled_segment = scaler_payload.transform(segment_data)
            window_data = scaled_segment.reshape(1, window_size, num_features)
            pred_probs = payload_model.predict(window_data)[0]
            pred_class_idx = np.argmax(pred_probs)
            class_map = {0: '0 kg', 1: '5 kg', 2: '10 kg', 3: '15 kg'}
        elif model_type == "Intention":
            scaled_segment = scaler_intention.transform(segment_data)
            window_data = scaled_segment.reshape(1, window_size, num_features)
            pred_probs = intention_model.predict(window_data)[0]
            pred_class_idx = np.argmax(pred_probs)
            class_map = {0: 'Still', 1: 'Move1', 2: 'Move2', 3: 'Move3'}
        else:
            return {"error": "Invalid model type selected"}

        prediction_label = class_map.get(pred_class_idx, "Unknown")
        confidence = float(pred_probs[pred_class_idx])
        probabilities = {class_map.get(i, f"Class {i}"): float(prob) for i, prob in enumerate(pred_probs)}
        return {
            "prediction": f"{model_type}: {prediction_label} (Confidence: {confidence:.2f})",
            "probabilities": probabilities
        }
    except Exception as e:
        return {"error": f"Error processing: {str(e)}"}

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# IMU Payload & Intention Classifier API")
    file_input = gr.File(label="Upload CSV File")
    model_dropdown = gr.Dropdown(choices=["Payload", "Intention"], label="Select Model")
    row_input = gr.Number(label=f"Starting Row Index (0 to Max-{window_size})", value=0)
    predict_button = gr.Button("Predict")
    output = gr.JSON(label="Prediction Result")
    predict_button.click(predict_from_upload, inputs=[file_input, model_dropdown, row_input], outputs=output)

demo.launch(share=False)  # Set share=False since we'll deploy to Hugging Face