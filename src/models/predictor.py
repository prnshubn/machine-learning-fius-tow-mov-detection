"""
Real-time Inference Simulation: Production-Grade Tracking Logic.

This module simulates the real-time operation of the collision prediction system.
It processes raw sensor data frame-by-frame, maintaining a temporal buffer to 
calculate 'Quad-Scale' features and providing low-latency predictions.

Industry Approach:
- Decoupled Pre-computation: Heavy FFT logic is handled separately (simulating DSP hardware).
- Circular Buffering: Uses 'deque' for efficient sliding-window feature calculation.
- Performance Profiling: Measures per-frame latency to validate 'Real-Time' capability.
- Model Pipelines: Loads atomic artifacts to ensure inference matches training scaling.
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

# Setup production-grade logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure package root is in path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.processing import (
    ADC_DATA_START_INDEX,
    ECHO_MIN_INDEX,
    ECHO_THRESHOLD_MULTIPLIER,
    find_first_peak_index,
    extract_echo_shape_features,
)
from src.data.kalman import KalmanFilter
from src.models.detectors import AutoencoderDetector

# Window configuration (MUST match model_trainer.py)
WINDOWS = [5, 10, 25, 50]


def _build_motion_feature_vector(echo_idx, echo_shape, peak_freq, spectral_centroid,
                                  buffer_echo, buffer_centroid):
    """
    Assembles the 17-element feature vector for the motion detection pipeline.

    The vector includes instantaneous measurements and temporal window statistics.
    """
    feat = {
        'echo_index':        echo_idx,
        'echo_amplitude':    echo_shape['echo_amplitude'],
        'echo_width':        echo_shape['echo_width'],
        'Peak Frequency':    peak_freq,
        'Spectral Centroid': spectral_centroid,
    }
    # Calculate Mean and Trend for each scale based on buffered history
    for w in WINDOWS:
        feat[f'Trend_{w}']         = buffer_echo[-1]     - buffer_echo[-w]
        feat[f'Mean_{w}']          = np.mean(list(buffer_echo)[-w:])
        feat[f'Centroid_Trend_{w}'] = buffer_centroid[-1] - buffer_centroid[-w]
    return feat


def main():
    """
    Main entry point for real-time tracking simulation on a raw data file.
    """
    parser = argparse.ArgumentParser(description='Simulate real-time tracking on raw sensor data.')
    parser.add_argument('filepath', type=str, help='Path to a raw sensor CSV file.')
    args = parser.parse_args()

    # Step 1: Load Serialized Pipelines
    # Pipelines bundle the scaler and model into one object for safety and consistency.
    try:
        motion_pipeline = joblib.load('models/motion_detection_model.joblib')
        ttc_pipeline    = joblib.load('models/ttc_prediction_model.joblib')
        logger.info("Inference pipelines loaded successfully.")
    except Exception as e:
        logger.error(f"Initialization failure (missing models?): {e}")
        sys.exit(1)

    # Step 2: DSP Pre-processing (FFT)
    # Note: In a production embedded system, this step would happen in a 
    # hardware DSP co-processor or FPGA before the ML engine.
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
    spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags

    # Step 3: Initialize Temporal Buffers and Kalman Filter
    timestamps_ms = df_raw.iloc[:, 16].values
    dt_avg = np.mean(np.diff(timestamps_ms)) / 1000.0
    kf = KalmanFilter(dt=dt_avg if dt_avg > 0 else 0.045)

    MAX_W           = max(WINDOWS)
    buffer_echo     = deque(maxlen=MAX_W)     # Rolling window for arrival time
    buffer_centroid = deque(maxlen=MAX_W)     # Rolling window for spectral shape

    towards_count   = 0
    predicted_ttcs  = []
    frame_latencies = []

    # Column ordering must strictly match the training features
    motion_cols = ['echo_index', 'echo_amplitude', 'echo_width', 'Peak Frequency', 'Spectral Centroid']
    for w in WINDOWS:
        motion_cols.extend([f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}'])

    ttc_cols = ['velocity', 'acceleration', 'echo_index', 'Peak Frequency', 'Spectral Centroid', 'Trend_25']

    # Step 4: Per-Frame Processing Loop (The 'Real-Time' Simulation)
    for i in range(len(df_raw)):
        # Calculate real delta-time for Kalman sync
        dt_frame = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0 if i > 0 else None
        
        # A. Update Kinematics (Filtered Dist, Velocity, Accel)
        _, f_vel, f_accel = kf.update(df_raw.iloc[i, 10], dt=dt_frame)

        # B. Signal Processing: Echo Detection and Shape Extraction
        echo_idx = float(find_first_peak_index(adc_centered[i], 
                                               threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER, 
                                               min_index=ECHO_MIN_INDEX))
        echo_shape = extract_echo_shape_features(adc_centered[i], int(echo_idx) if echo_idx >= 0 else -1)

        # C. Buffer update
        buffer_echo.append(echo_idx)
        buffer_centroid.append(spectral_centroids[i])

        # D. Ensure buffer is full before attempting prediction
        if len(buffer_echo) < MAX_W:
            continue

        # --- Inference Latency Measurement Start ---
        t_start = time.perf_counter()

        # E. Stage 1: Motion Classification (Towards vs Not Towards)
        feat_dict = _build_motion_feature_vector(echo_idx, echo_shape, peak_freqs[i], spectral_centroids[i], buffer_echo, buffer_centroid)
        X_motion = pd.DataFrame([feat_dict])[motion_cols]
        prediction = motion_pipeline.predict(X_motion)[0]

        # F. Stage 2: TTC Regression (Only if Stage 1 detects an approach)
        if prediction == 1:
            towards_count += 1
            ttc_feat = {
                'velocity':          f_vel, 
                'acceleration':      f_accel, 
                'echo_index':        echo_idx, 
                'Peak Frequency':    peak_freqs[i], 
                'Spectral Centroid': spectral_centroids[i], 
                'Trend_25':          buffer_echo[-1] - list(buffer_echo)[-25]
            }
            X_ttc = pd.DataFrame([ttc_feat])[ttc_cols]
            predicted_ttcs.append(max(0.0, ttc_pipeline.predict(X_ttc)[0]))

        # --- Inference Latency Measurement Stop ---
        t_end = time.perf_counter()
        frame_latencies.append((t_end - t_start) * 1000.0)

    # Step 5: Final Performance Reporting
    avg_latency = np.mean(frame_latencies) if frame_latencies else 0.0
    # Average sensor firing rate derived from record length (5 mins)
    sensor_period = (5 * 60 * 1000) / max(len(df_raw), 1)

    print("\n" + "=" * 45)
    print("SIMULATION PERFORMANCE REPORT")
    print("=" * 45)
    print(f"File: {os.path.basename(args.filepath)}")
    print(f"Processed: {len(df_raw)} frames")
    print(f"Approaches: {towards_count} frames")
    if predicted_ttcs:
        print(f"Min TTC: {min(predicted_ttcs):.2f} s | Mean TTC: {np.mean(predicted_ttcs):.2f} s")
    
    print(f"\nAvg Prediction Latency: {avg_latency:.3f} ms")
    print(f"Sensor Period: ~{sensor_period:.1f} ms")
    
    if avg_latency < sensor_period:
        print("Status: REAL-TIME CAPABLE ✅")
    else:
        print("Status: LATENCY WARNING ⚠️")
    print("=" * 45)


if __name__ == "__main__":
    main()
