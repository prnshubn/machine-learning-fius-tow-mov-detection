"""
Feature Engineering Module: Extracts raw sensor features from radar ADC data.

This module processes raw ADC windows into a multi-scale feature dataset. 
It implements 'Quad-Scale Temporal Windows' (Short, Mid, Long) to capture both 
instantaneous spikes and long-term trends in object movement.

Industry Approach:
- Modular extraction: Features are grouped by physical/spectral properties.
- Quad-Scale Windows: Captures movement dynamics across different time horizons.
- Shared Constants: Ensures consistency between training and inference data.
- Dynamic Blanking: Prevents blind spots by shrinking the Tx pulse rejection 
  window when kinematics indicate an object is extremely close.
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
    """
    # System Constants
    SAMPLING_RATE  = 1953125
    SPEED_OF_SOUND = 343.0
    WINDOWS        = [5, 10, 25, 50]  
    
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
        f"base echo blanking={ECHO_MIN_INDEX}, "
        f"sigma_threshold={ECHO_THRESHOLD_MULTIPLIER}..."
    )
    all_features_dfs = []

    for filepath in csv_files:
        filename     = os.path.basename(filepath)
        object_label = "_".join(filename.replace('.csv', '').split('_')[2:])

        try:
            df         = pd.read_csv(filepath, header=None)
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            
            # Step 1: Pre-processing (DC Centering)
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)

            # Step 2: Kinematics via Sensor Fusion (Kalman Filter)
            # Run this FIRST so we can use distance to solve the close-range blind spot
            f_distances, velocity, acceleration = apply_kalman_filter(
                df.iloc[:, 10], df.iloc[:, 16]
            )

            # Step 3: Per-frame Echo Detection (with Dynamic Blanking Zone)
            echo_indices = []
            for i, row in enumerate(adc_centered):
                expected_echo_idx = (f_distances[i] * 2.0 / SPEED_OF_SOUND) * SAMPLING_RATE
                
                if 0 < expected_echo_idx < (ECHO_MIN_INDEX + 10):
                    dynamic_min_index = max(5, int(expected_echo_idx) - 5)
                else:
                    dynamic_min_index = ECHO_MIN_INDEX

                idx = find_first_peak_index(
                    row,
                    threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER,
                    min_index=dynamic_min_index,
                )
                echo_indices.append(float(idx))
            
            echo_indices = np.array(echo_indices)

            # Step 4: Echo Shape Analysis (Amplitude and FWHM Width)
            echo_shapes     = [
                extract_echo_shape_features(row, int(idx) if idx >= 0 else -1)
                for row, idx in zip(adc_centered, echo_indices)
            ]
            echo_amplitudes = np.array([s['echo_amplitude'] for s in echo_shapes])
            echo_widths     = np.array([s['echo_width']     for s in echo_shapes])

            # Step 5: Spectral Transformation (Fast Fourier Transform)
            fft_results  = np.fft.fft(adc_centered, axis=1)
            n_samples    = adc_matrix.shape[1]
            magnitudes   = np.abs(fft_results[:, 1:n_samples // 2])
            frequencies  = np.fft.fftfreq(n_samples, d=1 / SAMPLING_RATE)[1:n_samples // 2]
            
            peak_freqs        = frequencies[np.argmax(magnitudes, axis=1)]
            spectral_centroids = (
                np.sum(frequencies * magnitudes, axis=1)
                / np.sum(magnitudes, axis=1)
            )

            # Step 6: Temporal Quad-Scale Windowing
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
                feature_dict[f'Trend_{w}']         = temp_df['echo'].diff(periods=w - 1).fillna(0).values
                feature_dict[f'Mean_{w}']           = temp_df['echo'].rolling(window=w, min_periods=1).mean().values
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