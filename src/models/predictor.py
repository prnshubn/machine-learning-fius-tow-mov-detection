"""
Real-time Inference Simulation: Production-Grade Tracking Logic.

This module simulates the real-time operation of the collision prediction system.
It dynamically calculates Quad-Scale features, predicts motion, TTC, and 
now utilizes a Material Classifier to calculate real-time Impact Force (F=ma).
"""

import os
import sys
import time
import pandas as pd
import joblib
import argparse
import numpy as np
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.processing import (
    ADC_DATA_START_INDEX, ECHO_MIN_INDEX, ECHO_THRESHOLD_MULTIPLIER,
    find_first_peak_index, extract_echo_shape_features,
)
from src.data.kalman import KalmanFilter
from src.models.detectors import AutoencoderDetector

WINDOWS = [5, 10, 25, 50]
SPEED_OF_SOUND = 343.0

# Mass definitions (in kg) for Force calculation (F=ma)
MASS_ESTIMATES = {
    'human': 70.0,
    'metal_plate': 2.0,
    'cardboard': 0.5,
}

def _build_motion_feature_vector(echo_idx, echo_shape, peak_freq, spectral_centroid, buffer_echo, buffer_centroid):
    feat = {
        'echo_index':        echo_idx,
        'echo_amplitude':    echo_shape['echo_amplitude'],
        'echo_width':        echo_shape['echo_width'],
        'Peak Frequency':    peak_freq,
        'Spectral Centroid': spectral_centroid,
    }
    for w in WINDOWS:
        idx = min(w, len(buffer_echo))
        feat[f'Trend_{w}']          = buffer_echo[-1]     - buffer_echo[-idx]
        feat[f'Mean_{w}']           = np.mean(list(buffer_echo)[-idx:])
        feat[f'Centroid_Trend_{w}'] = buffer_centroid[-1] - buffer_centroid[-idx]
    return feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to raw sensor CSV.')
    args = parser.parse_args()

    # Step 1: Load Serialized Pipelines
    try:
        motion_pipeline   = joblib.load('models/motion_detection_model.joblib')
        ttc_pipeline      = joblib.load('models/ttc_prediction_model.joblib')
        material_pipeline = joblib.load('models/material_classifier_model.joblib')
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    logger.info(f"Analyzing file: {os.path.basename(args.filepath)}")
    df_raw        = pd.read_csv(args.filepath, header=None)
    SAMPLING_RATE = 1953125

    adc_matrix   = df_raw.iloc[:, ADC_DATA_START_INDEX:].values
    adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)

    fft_results = np.fft.fft(adc_centered, axis=1)
    magnitudes  = np.abs(fft_results[:, 1:adc_matrix.shape[1] // 2])
    frequencies = np.fft.fftfreq(adc_matrix.shape[1], d=1 / SAMPLING_RATE)[1:adc_matrix.shape[1] // 2]

    peak_freqs         = frequencies[np.argmax(magnitudes, axis=1)]
    sum_mags           = np.sum(magnitudes, axis=1)
    
    # Avoid division by zero
    sum_mags[sum_mags == 0] = 1e-10 
    spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags

    timestamps_ms = df_raw.iloc[:, 16].values
    dt_avg = np.mean(np.diff(timestamps_ms)) / 1000.0
    kf = KalmanFilter(dt=dt_avg if dt_avg > 0 else 0.045)

    MAX_W           = max(WINDOWS)
    buffer_echo     = deque(maxlen=MAX_W)     
    buffer_centroid = deque(maxlen=MAX_W)     

    towards_count   = 0
    predicted_ttcs  = []
    predicted_forces = []
    detected_materials = []
    frame_latencies = []

    motion_cols = ['echo_index', 'echo_amplitude', 'echo_width', 'Peak Frequency', 'Spectral Centroid']
    for w in WINDOWS: motion_cols.extend([f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}'])
    ttc_cols = ['velocity', 'acceleration', 'echo_index', 'Peak Frequency', 'Spectral Centroid', 'Trend_25']
    mat_cols = ['echo_amplitude', 'echo_width', 'Spectral Centroid', 'Peak Frequency']

    for i in range(len(df_raw)):
        dt_frame = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0 if i > 0 else None
        f_dist, f_vel, f_accel = kf.update(df_raw.iloc[i, 10], dt=dt_frame)

        expected_echo_idx = (f_dist * 2.0 / SPEED_OF_SOUND) * SAMPLING_RATE
        dynamic_min_index = max(5, int(expected_echo_idx) - 5) if 0 < expected_echo_idx < (ECHO_MIN_INDEX + 10) else ECHO_MIN_INDEX

        echo_idx = float(find_first_peak_index(adc_centered[i], threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER, min_index=dynamic_min_index))
        echo_shape = extract_echo_shape_features(adc_centered[i], int(echo_idx) if echo_idx >= 0 else -1)

        buffer_echo.append(echo_idx)
        buffer_centroid.append(spectral_centroids[i])

        if len(buffer_echo) < MAX_W: continue

        t_start = time.perf_counter()

        # Stage 1: Motion Classification
        feat_dict = _build_motion_feature_vector(echo_idx, echo_shape, peak_freqs[i], spectral_centroids[i], buffer_echo, buffer_centroid)
        X_motion = pd.DataFrame([feat_dict])[motion_cols]
        prediction = motion_pipeline.predict(X_motion)[0]

        # Stage 2 & 3: TTC Regression & Force Estimation
        if prediction == 1:
            towards_count += 1
            
            # Predict TTC
            ttc_feat = {
                'velocity': f_vel, 'acceleration': f_accel, 'echo_index': echo_idx, 
                'Peak Frequency': peak_freqs[i], 'Spectral Centroid': spectral_centroids[i], 
                'Trend_25': buffer_echo[-1] - list(buffer_echo)[-min(25, len(buffer_echo))]
            }
            X_ttc = pd.DataFrame([ttc_feat])[ttc_cols]
            predicted_ttcs.append(max(0.0, ttc_pipeline.predict(X_ttc)[0]))

            # Predict Material & Force
            mat_feat = {
                'echo_amplitude': echo_shape['echo_amplitude'], 
                'echo_width': echo_shape['echo_width'], 
                'Spectral Centroid': spectral_centroids[i], 
                'Peak Frequency': peak_freqs[i]
            }
            X_mat = pd.DataFrame([mat_feat])[mat_cols]
            mat_pred = material_pipeline.predict(X_mat)[0]
            
            # Match predicted string to mass (fallback to 1.0kg if unknown)
            detected_mass = 1.0 
            for key, mass in MASS_ESTIMATES.items():
                if key in str(mat_pred).lower():
                    detected_mass = mass
                    break
                    
            force = detected_mass * abs(f_accel) # F = ma
            predicted_forces.append(force)
            detected_materials.append(mat_pred)

        t_end = time.perf_counter()
        frame_latencies.append((t_end - t_start) * 1000.0)

    # Final Reporting
    avg_latency = np.mean(frame_latencies) if frame_latencies else 0.0
    sensor_period = (5 * 60 * 1000) / max(len(df_raw), 1)

    print("\n" + "=" * 45)
    print("SIMULATION PERFORMANCE REPORT")
    print("=" * 45)
    print(f"File: {os.path.basename(args.filepath)}")
    print(f"Processed: {len(df_raw)} frames | Approaches: {towards_count} frames")
    if predicted_ttcs:
        print(f"Min TTC: {min(predicted_ttcs):.2f} s | Mean TTC: {np.mean(predicted_ttcs):.2f} s")
        
        # Output Force metrics
        top_material = max(set(detected_materials), key=detected_materials.count)
        print(f"Detected Material: {top_material}")
        print(f"Max Impact Force: {max(predicted_forces):.2f} N | Mean Force: {np.mean(predicted_forces):.2f} N")

    print(f"\nAvg Prediction Latency: {avg_latency:.3f} ms")
    print("Status: REAL-TIME CAPABLE ✅" if avg_latency < sensor_period else "Status: LATENCY WARNING ⚠️")
    print("=" * 45)

if __name__ == "__main__":
    main()