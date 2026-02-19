estimates Time-To-Collision (TTC), and generates a safety protocol.

"""
This script uses the trained model to make a prediction on a new, single data sample.
It reads a small sequence of data to calculate kinematics, predicts motion, 
estimates Time-To-Collision (TTC), and generates a safety protocol.

Main functionalities:
- Load a new sensor data file
- Extract features (spectral, kinematic, etc.)
- Use trained classifier to predict motion
- Use trained regression model to predict TTC
- Generate a safety protocol using LLM logic
"""

# Import required libraries
import os
import sys
import pandas as pd
import joblib
import argparse
import numpy as np

# Add the project root to the Python path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import feature extraction and safety protocol utilities
from src.data.processing import perform_fft, extract_spectral_features, find_first_peak_index, calculate_kinematics, ADC_DATA_START_INDEX
from src.models.llm_safety_module import generate_safety_protocol, print_safety_report


def predict_motion(data_row, model):
    """
    Predicts motion from a single row of sensor data features using the trained classifier.
    Args:
        data_row (pd.DataFrame): Feature row for prediction
        model: Trained classifier
    Returns:
        str: Predicted label or error message
    """
    try:
        prediction = model.predict(data_row)
        return prediction[0]
    except Exception as e:
        return f"Prediction error: {e}"


def predict_ttc(features, model, scaler):
    """
    Predicts Time-To-Collision (TTC) using the trained regression model and scaler.
    Args:
        features (pd.DataFrame): Feature row(s) for prediction
        model: Trained regression model
        scaler: Fitted scaler for feature normalization
    Returns:
        float or None: Predicted TTC value or None if error
    """
    try:
        # Select features expected by the model (order must match training)
        feature_cols = [
            'distance', 'velocity', 'acceleration', 
            'Peak Frequency', 'Mean Frequency', 'Spectral Centroid', 
            'Spectral Skewness', 'Spectral Kurtosis', 'first_peak_index'
        ]
        # Ensure features is a DataFrame with these columns
        X = features[feature_cols]
        # Scale features
        X_scaled = scaler.transform(X)
        # Predict TTC
        ttc = model.predict(X_scaled)
        return ttc[0]
    except Exception as e:
        print(f"TTC Prediction error: {e}")
        return None

def main():
    """
    Main function to load models and data, then make predictions and safety reports.
    Steps:
    1. Parse command-line arguments for input file
    2. Load trained models and scaler
    3. Load and preprocess input data
    4. Extract features
    5. Predict motion and TTC
    6. Generate and print safety protocol
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Predict motion and TTC from a sensor data file.')
    parser.add_argument('filepath', type=str, help='The path to the raw CSV data file.')
    args = parser.parse_args()

    # --- Load Models ---
    motion_model_path = 'models/motion_detection_model.joblib'
    ttc_model_path = 'models/ttc_model.joblib'
    ttc_scaler_path = 'models/ttc_scaler.joblib'
    
    if not os.path.exists(motion_model_path):
        print(f"Error: Motion model '{motion_model_path}' not found.")
        return

    print(f"Loading motion model from '{motion_model_path}'...")
    motion_model = joblib.load(motion_model_path)
    
    ttc_model = None
    ttc_scaler = None
    if os.path.exists(ttc_model_path) and os.path.exists(ttc_scaler_path):
        print(f"Loading TTC model and scaler...")
        ttc_model = joblib.load(ttc_model_path)
        ttc_scaler = joblib.load(ttc_scaler_path)
    else:
        print("Warning: TTC model or scaler not found. TTC prediction will be skipped.")

    # --- Load Data Sample (Sequence) ---
    if not os.path.exists(args.filepath):
        print(f"Error: The data file '{args.filepath}' was not found.")
        return
        
    # Read 5 rows to calculate kinematics
    N_ROWS = 5
    print(f"Loading last {N_ROWS} data points from '{args.filepath}' for context...")
    try:
        # Read the file
        df = pd.read_csv(args.filepath, header=None)
        if len(df) > N_ROWS:
            sample_df = df.iloc[:N_ROWS].copy() # Take first 5 for demo
        else:
            sample_df = df.copy()
            
    except Exception as e:
        print(f"An error occurred while reading the data file: {e}")
        return

    # --- Process Data ---
    # Extract distances and timestamps
    distances = sample_df.iloc[:, 10]
    timestamps = sample_df.iloc[:, 16]
    
    # Calculate Kinematics
    velocity, acceleration = calculate_kinematics(distances, timestamps)
    
    # We use the LAST row for prediction
    last_idx = len(sample_df) - 1
    last_row = sample_df.iloc[[last_idx]]
    
    # Perform FFT
    SAMPLING_RATE = 1953125
    frequencies, magnitudes = perform_fft(last_row, SAMPLING_RATE)
    features = extract_spectral_features(frequencies, magnitudes)
    
    if features is None:
        print("Error: Could not extract spectral features.")
        return
        
    # Add other features
    features['distance'] = distances.iloc[last_idx]
    features['velocity'] = velocity.iloc[last_idx]
    features['acceleration'] = acceleration.iloc[last_idx]
    
    adc_data = last_row.iloc[0, ADC_DATA_START_INDEX:].values.flatten()
    features['first_peak_index'] = find_first_peak_index(adc_data)
    
    # Prepare Feature DataFrame
    feature_df = pd.DataFrame([features])
    
    # --- Predict Motion ---
    predicted_motion = predict_motion(feature_df, motion_model)
    
    print(f"""
--- Motion Prediction ---
Prediction: {str(predicted_motion).upper()}
-------------------------
""")

    # --- Predict TTC & Safety ---
    if predicted_motion == 'approaching' and ttc_model is not None:
        ttc = predict_ttc(feature_df, ttc_model, ttc_scaler)
        
        if ttc is not None:
            # Generate Safety Report
            report = generate_safety_protocol(
                ttc=ttc, 
                velocity=features['velocity'], 
                acceleration=features['acceleration']
            )
            print_safety_report(report)
    elif predicted_motion == 'approaching' and ttc_model is None:
        print("Note: Approaching detected, but TTC model is not available.")
    else:
        print("Object is not approaching. No safety protocol required.")

if __name__ == "__main__":
    main()
