"""
Real-time Prediction Simulation: Uses PURE SENSOR DATA to detect approaches.
Implements QUAD-SCALE temporal windows (5, 10, 25, 50) for dynamic inference.
"""
import os
import sys
import pandas as pd
import joblib
import argparse
import numpy as np
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.processing import ADC_DATA_START_INDEX
from src.data.kalman import KalmanFilter

# Import shared detector logic
from src.models.detectors import AutoencoderDetector

def main():
    parser = argparse.ArgumentParser(description='Simulate tracking on pure sensor data.')
    parser.add_argument('filepath', type=str, help='Path to raw sensor CSV.')
    args = parser.parse_args()

    # 1. Load artifacts
    try:
        motion_model = joblib.load('models/motion_detection_model.joblib')
        motion_scaler = joblib.load('models/motion_scaler.joblib')
        ttc_model = joblib.load('models/ttc_prediction_model.joblib')
        ttc_scaler = joblib.load('models/ttc_scaler.joblib')
        logger.info(f"Loaded Quad-Scale Models.")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

    # 2. Pre-calculation
    logger.info(f"Analyzing file: {os.path.basename(args.filepath)}")
    df_raw = pd.read_csv(args.filepath, header=None)
    SAMPLING_RATE = 1953125
    adc_matrix = df_raw.iloc[:, ADC_DATA_START_INDEX:].values
    adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)
    
    fft_results = np.fft.fft(adc_centered, axis=1)
    magnitudes = np.abs(fft_results[:, 1:adc_matrix.shape[1]//2])
    frequencies = np.fft.fftfreq(adc_matrix.shape[1], d=1/SAMPLING_RATE)[1:adc_matrix.shape[1]//2]
    
    peak_freqs = frequencies[np.argmax(magnitudes, axis=1)]
    sum_mags = np.sum(magnitudes, axis=1)
    spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags
    
    # 3. Quad-Scale Stream Simulation
    WINDOWS = [5, 10, 25, 50]
    MAX_W = max(WINDOWS)
    buffer_echo = deque(maxlen=MAX_W)
    buffer_centroid = deque(maxlen=MAX_W)
    kf = KalmanFilter(dt=0.045)
    
    towards_count = 0
    predicted_ttcs = []

    for i in range(len(df_raw)):
        _, f_vel, f_accel = kf.update(df_raw.iloc[i, 10])
        
        # Peak Detection
        row_adc = adc_centered[i]
        threshold = 4 * np.std(row_adc)
        hits = np.where(np.abs(row_adc) > threshold)[0]
        valid_hits = hits[hits > 20]
        echo_idx = float(valid_hits[0]) if len(valid_hits) > 0 else -1.0
        
        buffer_echo.append(echo_idx)
        buffer_centroid.append(spectral_centroids[i])
        
        if len(buffer_echo) < MAX_W:
            continue

        # Quad-Scale Feature Set
        feat_dict = {
            'echo_index': echo_idx,
            'Peak Frequency': peak_freqs[i],
            'Spectral Centroid': spectral_centroids[i]
        }
        
        for w in WINDOWS:
            feat_dict[f'Trend_{w}'] = buffer_echo[-1] - buffer_echo[-w]
            feat_dict[f'Mean_{w}'] = np.mean(list(buffer_echo)[-w:])
            feat_dict[f'Centroid_Trend_{w}'] = buffer_centroid[-1] - buffer_centroid[-w]
        
        # Classification
        cols = ['echo_index', 'Peak Frequency', 'Spectral Centroid']
        for w in WINDOWS:
            cols.extend([f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}'])
        
        X_motion = motion_scaler.transform(pd.DataFrame([feat_dict])[cols])
        
        if motion_model.predict(X_motion)[0] == 1:
            towards_count += 1
            
            # Regression (uses fixed features from its own training)
            ttc_feat = {
                'velocity': f_vel,
                'acceleration': f_accel,
                'echo_index': echo_idx,
                'Peak Frequency': peak_freqs[i],
                'Spectral Centroid': spectral_centroids[i],
                'Trend_50': buffer_echo[-1] - buffer_echo[0]
            }
            t_cols = ['velocity', 'acceleration', 'echo_index', 'Peak Frequency', 'Spectral Centroid', 'Trend_50']
            X_ttc = ttc_scaler.transform(pd.DataFrame([ttc_feat])[t_cols])
            predicted_ttcs.append(max(0.0, ttc_model.predict(X_ttc)[0]))

    # 4. Final summary
    logger.info(f"Analysis complete.")
    print("\n" + "="*40 + "\nQUAD-SCALE DETECTION REPORT\n" + "="*40)
    print(f"Towards Frames: {towards_count}")
    if predicted_ttcs:
        print(f"Min TTC: {min(predicted_ttcs):.2f}s")
    print(f"Movement Presence: {'YES' if towards_count > 15 else 'NO'}")
    print("="*40)

if __name__ == "__main__":
    main()
