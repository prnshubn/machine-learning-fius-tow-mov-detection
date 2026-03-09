"""
This script simulates real-time deployment by streaming through a raw data file.
It detects distinct "Encounters" (every time you approached the sensor) 
and generates a full safety incident report (saved to reports/).
"""

import os
import sys
import pandas as pd
import joblib
import argparse
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.processing import perform_fft, extract_spectral_features, find_first_peak_index, calculate_kinematics, ADC_DATA_START_INDEX
from src.models.llm_safety_module import generate_safety_protocol

def main():
    parser = argparse.ArgumentParser(description='Simulate real-time tracking on a sensor data file.')
    parser.add_argument('filepath', type=str, help='Path to the raw CSV data file.')
    args = parser.parse_args()

    # --- Load Models ---
    try:
        motion_model = joblib.load('models/motion_detection_model.joblib')
        motion_scaler = joblib.load('models/motion_scaler.joblib')
        ttc_model = joblib.load('models/ttc_model.joblib')
        ttc_scaler = joblib.load('models/ttc_scaler.joblib')
    except Exception as e:
        print(f"Error loading models: {e}.")
        sys.exit(1)

    input_filename = os.path.basename(args.filepath)
    print(f"\n{'='*60}")
    print(f" SESSION SAFETY ANALYSIS: {input_filename} ".center(60, '='))
    print(f"{'='*60}\n")

    # --- Load Data Meta ---
    df_meta = pd.read_csv(args.filepath, header=None, usecols=[10, 16])
    df_meta.columns = ['distance', 'timestamp']
    velocity, acceleration = calculate_kinematics(df_meta['distance'], df_meta['timestamp'])
    
    encounters = []
    current_encounter = []
    
    # Heuristic: If we don't see movement for 20 frames, the encounter has ended
    GAP_THRESHOLD = 20 
    gap_counter = 0

    print("Streaming data and analyzing motion patterns...")

    for i in range(5, len(df_meta)):
        # Lowered threshold from -0.15 to -0.01 based on actual data distribution
        if velocity[i] < -0.01: 
            gap_counter = 0
            
            # Deep Feature Extraction
            row_data = pd.read_csv(args.filepath, header=None, skiprows=i, nrows=1)
            freqs, mags = perform_fft(row_data, 1953125)
            features = extract_spectral_features(freqs, mags)
            
            if features:
                features['distance'] = df_meta.loc[i, 'distance']
                features['velocity'] = velocity[i]
                features['acceleration'] = acceleration[i]
                features['first_peak_index'] = find_first_peak_index(row_data.iloc[0, ADC_DATA_START_INDEX:].values)
                
                # Motion Verification
                feat_cols = ['Peak Frequency', 'Mean Frequency', 'Spectral Centroid', 'Spectral Skewness', 'Spectral Kurtosis', 'distance', 'first_peak_index']
                X_motion = motion_scaler.transform(pd.DataFrame([features])[feat_cols])
                if motion_model.predict(X_motion)[0] == 'approaching':
                    
                    # TTC Calculation
                    X_ttc = ttc_scaler.transform(pd.DataFrame([features])[['distance', 'velocity', 'acceleration', 'Peak Frequency', 'Mean Frequency', 'Spectral Centroid', 'Spectral Skewness', 'Spectral Kurtosis', 'first_peak_index']])
                    ttc = ttc_model.predict(X_ttc)[0]
                    
                    current_encounter.append({
                        'dist': features['distance'],
                        'vel': features['velocity'],
                        'ttc': ttc
                    })
        else:
            gap_counter += 1
            if gap_counter >= GAP_THRESHOLD and current_encounter:
                best_ttc = min(e['ttc'] for e in current_encounter)
                max_vel = min(e['vel'] for e in current_encounter)
                encounters.append({
                    'Encounter_ID': len(encounters) + 1,
                    'Peak_Speed': abs(max_vel),
                    'Min_TTC': best_ttc,
                    'Risk_Level': "CRITICAL" if best_ttc < 1.0 else "HIGH" if best_ttc < 3.0 else "MEDIUM"
                })
                current_encounter = []

    # Final wrap up
    if current_encounter:
        best_ttc = min(e['ttc'] for e in current_encounter)
        encounters.append({
            'Encounter_ID': len(encounters) + 1,
            'Peak_Speed': abs(min(e['vel'] for e in current_encounter)),
            'Min_TTC': best_ttc,
            'Risk_Level': "CRITICAL" if best_ttc < 1.0 else "HIGH" if best_ttc < 3.0 else "MEDIUM"
        })

    # --- FINAL REPORT ---
    if not encounters:
        print("\nResult: No safety incidents detected. User remained at a safe distance.")
    else:
        # Create DataFrame for the report
        report_df = pd.DataFrame(encounters)
        
        # Save to CSV
        output_path = f"reports/safety_log_{input_filename}"
        os.makedirs('reports', exist_ok=True)
        report_df.to_csv(output_path, index=False)
        
        print(f"\nDETECTION SUMMARY (Saved to: {output_path})")
        print(f"{'-'*60}")
        print(report_df.to_string(index=False))
        print(f"{'-'*60}")
        print(f"\nTotal Encounters Neutralized: {len(encounters)}")
        
        # Detailed protocol for worst case
        worst = report_df.loc[report_df['Min_TTC'].idxmin()]
        print(f"\n--- Protocol for Most Critical Encounter (TTC: {worst['Min_TTC']:.2f}s) ---")
        report = generate_safety_protocol(worst['Min_TTC'], -worst['Peak_Speed'], 0)
        print(f"ACTION REQUIRED: {report['recommended_protocol']}")
        print(f"ASSESSMENT: {report['assessment']}")

if __name__ == "__main__":
    main()
