"""
Visualization Module: Raw Echo Peak Detection

Generates a high-resolution plot of a single ADC frame to visually validate 
the adaptive noise floor and first-peak detection logic. This directly addresses 
the requirement to demonstrate and prove echo detection capabilities.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure package root is in path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.processing import (
    ADC_DATA_START_INDEX,
    ECHO_MIN_INDEX,
    ECHO_THRESHOLD_MULTIPLIER,
    find_first_peak_index
)

def plot_single_echo(filepath, frame_idx=50, output_dir='reports/figures'):
    """
    Plots a single frame of ADC data, the dynamic noise threshold, and the detected peak.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(filepath, header=None)
    adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
    
    if frame_idx >= len(adc_matrix):
        print(f"  Warning: Frame index {frame_idx} out of bounds. Using frame 0.")
        frame_idx = 0
        
    # Select a single frame and center it (matching feature_extractor.py)
    raw_signal = adc_matrix[frame_idx]
    centered_signal = raw_signal - np.mean(raw_signal)
    
    # Calculate noise floor dynamically (matching processing.py)
    if ECHO_MIN_INDEX > 0 and ECHO_MIN_INDEX < len(centered_signal):
        noise_region = centered_signal[ECHO_MIN_INDEX:]
    else:
        noise_region = centered_signal
        
    std_val = np.std(noise_region)
    threshold = ECHO_THRESHOLD_MULTIPLIER * std_val
    
    # Detect the peak
    peak_idx = find_first_peak_index(centered_signal, ECHO_THRESHOLD_MULTIPLIER, ECHO_MIN_INDEX)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot the main signal
    plt.plot(centered_signal, label='DC-Centered ADC Signal', color='steelblue', linewidth=1.5)
    
    # Draw threshold lines
    plt.axhline(threshold, color='darkorange', linestyle='--', label=f'+{ECHO_THRESHOLD_MULTIPLIER}$\\sigma$ Noise Threshold')
    plt.axhline(-threshold, color='darkorange', linestyle='--')
    
    # Draw Tx blanking zone
    plt.axvspan(0, ECHO_MIN_INDEX, color='gray', alpha=0.2, label='Tx Pulse Blanking Zone')
    
    # Mark the detected peak
    if peak_idx != -1:
        plt.plot(peak_idx, centered_signal[peak_idx], 'ro', markersize=8, label=f'Detected First Peak (Index: {peak_idx})')
    else:
        plt.title(f'Ultrasonic Echo Detection (Frame {frame_idx}) - NO PEAK FOUND', color='red')
        
    if peak_idx != -1:
        plt.title(f'Ultrasonic Echo Detection (Frame {frame_idx})')
        
    plt.xlabel('ADC Sample Index')
    plt.ylabel('Amplitude (Voltage)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    out_path = os.path.join(output_dir, 'raw_echo_detection.png')
    plt.savefig(out_path, dpi=300)
    print(f"  Saved raw echo visualization to {out_path}")
    plt.close()

if __name__ == "__main__":
    raw_dir = 'data/raw'
    # Find the first available CSV file to use as a sample
    if os.path.exists(raw_dir):
        sample_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if sample_files:
            sample_file = os.path.join(raw_dir, sample_files[0])
            print(f"  Generating echo plot using sample file: {sample_file}")
            plot_single_echo(sample_file)
        else:
            print("  Warning: No raw CSV files found in data/raw/ to plot.")
    else:
        print(f"  Warning: Directory {raw_dir} does not exist.")