"""
Feature Engineering Module: Extracts raw sensor features from radar ADC data.

This module processes raw ADC windows into a multi-scale feature dataset. 
It implements 'Quad-Scale Temporal Windows' (Short, Mid, Long) to capture both 
instantaneous spikes and long-term trends in object movement.

Industry Approach:
- Modular extraction: Features are grouped by physical/spectral properties.
- Quad-Scale Windows: Captures movement dynamics across different time horizons.
- Shared Constants: Ensures consistency between training and inference data.
"""

import os
import glob
import sys
import pandas as pd
import numpy as np
import logging
from src.data.processing import (
    ADC_DATA_START_INDEX,
    ECHO_MIN_INDEX,
    ECHO_THRESHOLD_MULTIPLIER,
    find_first_peak_index,
    extract_echo_shape_features,
)
from src.data.kalman import apply_kalman_filter

# Standard industry logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_feature_dataset():
    """
    Main entry point: Orchestrates feature extraction across all raw CSV files.

    Pipeline:
    1. Scan data/raw/ for CSV files.
    2. For each file, compute per-row ADC features (Echo, Spectral, Shape).
    3. Apply Kalman Filter across the session to derive smooth kinematics.
    4. Apply Quad-Scale rolling windows to capture temporal dynamics.
    5. Aggregate and save to data/processed/features.csv.
    """
    # System Constants
    SAMPLING_RATE = 1953125
    WINDOWS       = [5, 10, 25, 50]  # Window sizes for Quad-Scale analysis
    
    raw_data_path       = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath     = os.path.join(processed_data_path, 'features.csv')

    os.makedirs(processed_data_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))

    if not csv_files:
        logger.error("No raw data files found in data/raw/")
        sys.exit(1)

    logger.info(
        f"Starting Quad-Scale extraction with windows {WINDOWS}, "
        f"echo blanking={ECHO_MIN_INDEX}, "
        f"sigma_threshold={ECHO_THRESHOLD_MULTIPLIER}..."
    )
    all_features_dfs = []

    for filepath in csv_files:
        filename     = os.path.basename(filepath)
        # Parse session label from filename (e.g., 'signal_1500_metal_plate.csv' -> 'metal_plate')
        object_label = "_".join(filename.replace('.csv', '').split('_')[2:])

        try:
            df         = pd.read_csv(filepath, header=None)
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            
            # Step 1: Pre-processing (DC Centering)
            # Center each row independently to normalize signal bias across time
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)

            # Step 2: Per-frame Echo Detection
            # find_first_peak_index handles adaptive noise floor estimation
            echo_indices = np.array([
                find_first_peak_index(
                    row,
                    threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER,
                    min_index=ECHO_MIN_INDEX,
                )
                for row in adc_centered
            ], dtype=float)

            # Step 3: Echo Shape Analysis (Amplitude and FWHM Width)
            echo_shapes     = [
                extract_echo_shape_features(row, int(idx) if idx >= 0 else -1)
                for row, idx in zip(adc_centered, echo_indices)
            ]
            echo_amplitudes = np.array([s['echo_amplitude'] for s in echo_shapes])
            echo_widths     = np.array([s['echo_width']     for s in echo_shapes])

            # Step 4: Spectral Transformation (Fast Fourier Transform)
            fft_results  = np.fft.fft(adc_centered, axis=1)
            n_samples    = adc_matrix.shape[1]
            magnitudes   = np.abs(fft_results[:, 1:n_samples // 2])
            frequencies  = np.fft.fftfreq(n_samples, d=1 / SAMPLING_RATE)[1:n_samples // 2]
            
            # Identify dominant frequency (Doppler shift) and Spectral Centroid
            peak_freqs        = frequencies[np.argmax(magnitudes, axis=1)]
            spectral_centroids = (
                np.sum(frequencies * magnitudes, axis=1)
                / np.sum(magnitudes, axis=1)
            )

            # Step 5: Kinematics via Sensor Fusion (Kalman Filter)
            # Inputs: Distance col 10, Timestamp col 16
            _, velocity, acceleration = apply_kalman_filter(
                df.iloc[:, 10], df.iloc[:, 16]
            )

            # Step 6: Temporal Quad-Scale Windowing
            # Rolling windows capture the stability (Mean) and rate of change (Trend)
            temp_df = pd.DataFrame({
                'echo':     echo_indices,
                'centroid': spectral_centroids,
            })

            # Base feature dictionary
            feature_dict = {
                'label':             object_label,
                'timestamp':         df.iloc[:, 16].values,
                'velocity':          velocity,
                'acceleration':      acceleration,
                'echo_index':        echo_indices,
                'echo_amplitude':    echo_amplitudes,
                'echo_width':        echo_widths,
                'Peak Frequency':    peak_freqs,
                'Spectral Centroid': spectral_centroids,
            }

            # Generate rolling features for each scale in {5, 10, 25, 50}
            for w in WINDOWS:
                # Trend: Absolute change over the window
                feature_dict[f'Trend_{w}']         = temp_df['echo'].diff(periods=w - 1).fillna(0).values
                # Mean: Average position to smooth jitter
                feature_dict[f'Mean_{w}']           = temp_df['echo'].rolling(window=w, min_periods=1).mean().values
                # Centroid Trend: Change in spectral shape (Doppler dynamics)
                feature_dict[f'Centroid_Trend_{w}'] = temp_df['centroid'].diff(periods=w - 1).fillna(0).values

            all_features_dfs.append(pd.DataFrame(feature_dict))
            logger.info(f"  Processed: {filename} ({len(df)} rows, label='{object_label}')")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    # Step 7: Final Consolidation and Persistence
    if not all_features_dfs:
        logger.error("No feature data was produced.")
        sys.exit(1)

    final_dataset = pd.concat(all_features_dfs, ignore_index=True)
    final_dataset.to_csv(output_filepath, index=False)
    logger.info(f"Feature dataset saved to {output_filepath}")


if __name__ == "__main__":
    build_feature_dataset()
