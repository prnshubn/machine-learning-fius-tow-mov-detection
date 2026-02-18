"""
This script builds a complete feature dataset from the raw sensor data.
It iterates through all the raw CSV files, extracts spectral features using the functions
from 'processing.py', and combines them into a single labeled dataset.

"""
import os
import glob
import pandas as pd
from .processing import perform_fft, extract_spectral_features

def build_feature_dataset():
    """
    Processes all raw data files to create a feature dataset in a memory-efficient way.
    """
    # --- Critical Step: Define Your Labels ---
    labels = {
        "signal_1500_metal_plate.csv": "metal_plate",
        "signal_1500_people_with_keeping_distance.csv": "people_distant",
        "signal_2000_cardboard.csv": "cardboard",
        "signal_2000_people.csv": "people_close",
    }
    
    # --- Parameters ---
    SAMPLING_RATE = 1953125
    
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath = os.path.join(processed_data_path, 'features.csv')
    
    os.makedirs(processed_data_path, exist_ok=True)
    
    # --- Data Processing Loop ---
    all_rows_features = []
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{raw_data_path}'.")
        return
        
    print(f"Found {len(csv_files)} files to process...")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        if filename not in labels:
            print(f"Warning: No label found for '{filename}'. Skipping this file.")
            continue
            
        print(f"Processing '{filename}'...")
        
        try:
            # Process the full file at once
            df = pd.read_csv(filepath, header=None)
            for i, row in df.iterrows():
                row_df = pd.DataFrame([row])
                # Extract distance from column 10
                distance = row_df.iloc[0, 10]
                
                # Perform FFT and extract spectral features
                frequencies, magnitudes = perform_fft(row_df, SAMPLING_RATE)
                features = extract_spectral_features(frequencies, magnitudes)
                
                if features is None:
                    print(f"Warning: Could not extract features from row {i} in '{filename}'.")
                    continue
                    
                # Add the distance and the label
                features['distance'] = distance
                features['label'] = labels[filename]
                all_rows_features.append(features)

        except Exception as e:
            print(f"An error occurred while processing '{filename}': {e}")
            continue # Move to the next file
            
    if not all_rows_features:
        print("No features were extracted. The process has been aborted.")
        return

    # --- Save the Dataset ---
    feature_dataset = pd.DataFrame(all_rows_features)
    
    cols = ['label'] + [col for col in feature_dataset.columns if col != 'label']
    feature_dataset = feature_dataset[cols]
    
    try:
        feature_dataset.to_csv(output_filepath, index=False)
        print(f"""
Successfully built the dataset!
- Processed {len(all_rows_features)} total measurements (rows).
- The feature dataset has been saved to: '{output_filepath}'""")
    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")

if __name__ == "__main__":
    build_feature_dataset()
