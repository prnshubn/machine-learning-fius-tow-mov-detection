"""
Feature Engineering Module: Extracts raw sensor features from radar ADC data.
Uses QUAD-SCALE TEMPORAL WINDOWS (5, 10, 25, 50) for robust pattern recognition.
"""
import os
import glob
import sys
import pandas as pd
import numpy as np
import logging
from scipy.stats import skew, kurtosis
from src.data.processing import ADC_DATA_START_INDEX, find_first_peak_index
from src.data.kalman import apply_kalman_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_feature_dataset():
    """
    Extracts features using Short (5, 10), Mid (25), and Long (50) windows.
    This captures sudden twitches, normal approaches, and slow drifts.
    """
    SAMPLING_RATE = 1953125
    WINDOWS = [5, 10, 25, 50]
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath = os.path.join(processed_data_path, 'features.csv')
    
    os.makedirs(processed_data_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    
    if not csv_files:
        logger.error(f"No raw data files found.")
        sys.exit(1)

    logger.info(f"Starting Quad-Scale extraction {WINDOWS} frames...")
    all_features_dfs = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        object_label = "_".join(filename.replace('.csv', '').split('_')[2:])
        
        try:
            df = pd.read_csv(filepath, header=None)
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)
            
            # 1. Echo Detection (Using Professor's requested function)
            echo_indices = []
            for row in adc_centered:
                idx = find_first_peak_index(row, threshold_multiplier=4)
                echo_indices.append(idx)
            echo_indices = np.array(echo_indices).astype(float)

            # 2. Spectral Features
            fft_results = np.fft.fft(adc_centered, axis=1)
            n_samples = adc_matrix.shape[1]
            magnitudes = np.abs(fft_results[:, 1:n_samples//2])
            frequencies = np.fft.fftfreq(n_samples, d=1/SAMPLING_RATE)[1:n_samples//2]
            peak_freqs = frequencies[np.argmax(magnitudes, axis=1)]
            spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / np.sum(magnitudes, axis=1)
            
            # 3. Kinematics (Ground Truth)
            _, velocity, acceleration = apply_kalman_filter(df.iloc[:, 10], df.iloc[:, 16])

            # 4. Temporal Trend Calculation
            temp_df = pd.DataFrame({'echo': echo_indices, 'centroid': spectral_centroids})
            
            feature_dict = {
                'label': object_label,
                'timestamp': df.iloc[:, 16],
                'velocity': velocity,
                'acceleration': acceleration,
                'echo_index': echo_indices,
                'Peak Frequency': peak_freqs,
                'Spectral Centroid': spectral_centroids
            }

            # Add dynamic windows
            for w in WINDOWS:
                feature_dict[f'Trend_{w}'] = temp_df['echo'].diff(periods=w-1).fillna(0)
                feature_dict[f'Mean_{w}'] = temp_df['echo'].rolling(window=w, min_periods=1).mean()
                feature_dict[f'Centroid_Trend_{w}'] = temp_df['centroid'].diff(periods=w-1).fillna(0)

            all_features_dfs.append(pd.DataFrame(feature_dict))

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    final_dataset = pd.concat(all_features_dfs, ignore_index=True)
    final_dataset.to_csv(output_filepath, index=False)
    logger.info(f"Feature dataset built with Quad-Scale context.")

if __name__ == "__main__":
    build_feature_dataset()
