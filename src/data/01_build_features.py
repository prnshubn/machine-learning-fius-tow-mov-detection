"""
This script builds a complete feature dataset from all raw sensor data files 
found in 'data/raw/'. It automatically infers labels from filenames 
following a strict naming convention.
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from src.data.processing import calculate_kinematics, find_first_peak_index, ADC_DATA_START_INDEX

def build_feature_dataset():
    """
    Processes all raw data files in 'data/raw/' using vectorized operations.
    Strict Naming Convention: signal_{distance}_{object_name}.csv
    Example: signal_1500_metal_plate.csv -> Label: metal_plate
    """
    SAMPLING_RATE = 1953125
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath = os.path.join(processed_data_path, 'features.csv')
    
    os.makedirs(processed_data_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{raw_data_path}'.")
        return
        
    all_features_dfs = []
    
    print(f"Scanning '{raw_data_path}' for data files...")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        # --- Strict Naming Convention Check ---
        # Expected: signal_{distance}_{object}.csv
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) < 3 or parts[0] != 'signal':
            print(f"Skipping '{filename}': Does not follow convention 'signal_DIST_OBJECT.csv'")
            continue
            
        # Extract object name (everything after the second underscore)
        object_label = "_".join(parts[2:])
        
        print(f"Processing '{filename}' -> Object Label: {object_label}")
        
        try:
            # Load the file
            df = pd.read_csv(filepath, header=None)
            
            # --- 1. Kinematics ---
            distances = df.iloc[:, 10]
            timestamps = df.iloc[:, 16]
            velocity, acceleration = calculate_kinematics(distances, timestamps)
            
            # --- 2. Spectral Features (Vectorized FFT) ---
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            n_rows, n_samples = adc_matrix.shape
            
            # Remove DC Offset
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)
            
            # Perform FFT
            fft_results = np.fft.fft(adc_centered, axis=1)
            magnitudes = np.abs(fft_results[:, 1:n_samples//2])
            frequencies = np.fft.fftfreq(n_samples, d=1/SAMPLING_RATE)[1:n_samples//2]
            
            # Extract Features
            peak_indices = np.argmax(magnitudes, axis=1)
            peak_freqs = frequencies[peak_indices]
            sum_mags = np.sum(magnitudes, axis=1)
            mean_freqs = np.sum(frequencies * magnitudes, axis=1) / sum_mags
            spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags
            spectral_skewness = skew(magnitudes, axis=1)
            spectral_kurtosis = kurtosis(magnitudes, axis=1)
            
            # --- 3. First Peak Detection ---
            row_means = np.mean(adc_matrix, axis=1, keepdims=True)
            row_stds = np.std(adc_matrix, axis=1, keepdims=True)
            thresholds = row_means + (3 * row_stds)
            mask = adc_matrix > thresholds
            first_peak_indices = np.argmax(mask, axis=1)
            first_peak_indices[~np.any(mask, axis=1)] = -1

            # --- 4. Combine ---
            file_features = pd.DataFrame({
                'label': object_label, # Dynamic label from filename
                'Peak Frequency': peak_freqs,
                'Mean Frequency': mean_freqs,
                'Spectral Centroid': spectral_centroids,
                'Spectral Skewness': spectral_skewness,
                'Spectral Kurtosis': spectral_kurtosis,
                'distance': distances,
                'timestamp': timestamps,
                'velocity': velocity,
                'acceleration': acceleration,
                'first_peak_index': first_peak_indices
            })
            
            all_features_dfs.append(file_features)

        except Exception as e:
            print(f"Error processing '{filename}': {e}")
            continue

    if not all_features_dfs:
        print("No valid data processed.")
        return

    final_dataset = pd.concat(all_features_dfs, ignore_index=True)
    final_dataset.to_csv(output_filepath, index=False)
    print(f"\nSuccessfully built dataset with {len(final_dataset)} rows from {len(all_features_dfs)} files.")
