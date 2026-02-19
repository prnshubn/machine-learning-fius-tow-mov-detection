"""
This script builds a complete feature dataset from the raw sensor data using 
high-performance vectorized operations. It avoids row-by-row loops (iterrows) 
to process large files in seconds.
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from src.data.processing import calculate_kinematics, find_first_peak_index, ADC_DATA_START_INDEX

def build_feature_dataset():
    """
    Processes all raw data files using vectorized NumPy operations.
    """
    labels = {
        "signal_1500_metal_plate.csv": "metal_plate",
        "signal_1500_people_with_keeping_distance.csv": "people_distant",
        "signal_2000_cardboard.csv": "cardboard",
        "signal_2000_people.csv": "people_close",
    }
    
    SAMPLING_RATE = 1953125
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath = os.path.join(processed_data_path, 'features.csv')
    
    os.makedirs(processed_data_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{raw_data_path}'.")
        return
        
    all_features_dfs = []
    
    print(f"Found {len(csv_files)} files to process...")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        if filename not in labels:
            continue
            
        print(f"Processing '{filename}' (Vectorized)...")
        
        try:
            # Load the entire file
            df = pd.read_csv(filepath, header=None)
            
            # --- 1. Kinematics (Already vectorized) ---
            distances = df.iloc[:, 10]
            timestamps = df.iloc[:, 16]
            velocity, acceleration = calculate_kinematics(distances, timestamps)
            
            # --- 2. Spectral Features (Vectorized FFT) ---
            # Extract only the ADC signal columns (from index 17 onwards)
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            n_rows, n_samples = adc_matrix.shape
            
            # Remove DC Offset (Subtract mean of each row)
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)
            
            # Perform FFT on all rows at once
            fft_results = np.fft.fft(adc_centered, axis=1)
            magnitudes = np.abs(fft_results[:, 1:n_samples//2]) # Positive frequencies, skip DC
            
            # Generate frequency axis
            frequencies = np.fft.fftfreq(n_samples, d=1/SAMPLING_RATE)[1:n_samples//2]
            
            # Calculate Spectral Features using vectorized NumPy
            peak_indices = np.argmax(magnitudes, axis=1)
            peak_freqs = frequencies[peak_indices]
            
            # Weighted average for Mean Frequency and Centroid
            sum_mags = np.sum(magnitudes, axis=1)
            mean_freqs = np.sum(frequencies * magnitudes, axis=1) / sum_mags
            spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags
            
            # Skewness and Kurtosis across axis 1
            spectral_skewness = skew(magnitudes, axis=1)
            spectral_kurtosis = kurtosis(magnitudes, axis=1)
            
            # --- 3. First Peak Detection (Vectorized) ---
            row_means = np.mean(adc_matrix, axis=1, keepdims=True)
            row_stds = np.std(adc_matrix, axis=1, keepdims=True)
            thresholds = row_means + (3 * row_stds)
            
            # Find the first index in each row where value > threshold
            mask = adc_matrix > thresholds
            # np.argmax on a boolean array returns the first index of 'True'
            first_peak_indices = np.argmax(mask, axis=1)
            # If no True found, argmax returns 0. Check if the value at index 0 is actually > threshold
            first_peak_indices[~np.any(mask, axis=1)] = -1

            # --- 4. Combine into a DataFrame ---
            file_features = pd.DataFrame({
                'label': labels[filename],
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
        print("No features extracted.")
        return

    # Combine all file dataframes into one
    final_dataset = pd.concat(all_features_dfs, ignore_index=True)
    
    try:
        final_dataset.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully built dataset with {len(final_dataset)} rows.")
        print(f"Saved to: '{output_filepath}'")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    build_feature_dataset()
