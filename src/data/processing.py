"""
This script provides a framework for processing ultrasonic sensor data to detect moving objects.
It includes functions to load data, perform a Fast Fourier Transform (FFT) to analyze the frequency spectrum,
and extract relevant features from the spectrum. These features can then be used to train a machine learning model.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew

# --- Constants ---
SENSOR_FREQUENCY = 40000  # 40 kHz, the frequency of the emitted pulse
ADC_DATA_START_INDEX = 17 # The starting index of the raw ADC data in the CSV file

def load_data(filepath, nrows=None):
    """
    Load data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        nrows (int, optional): The number of rows to read. Defaults to None (read all rows).

    Returns:
        pandas.DataFrame: The loaded data.
    """
    try:
        return pd.read_csv(filepath, nrows=nrows)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def perform_fft(data_frame, sampling_rate):
    """
    Perform a Fast Fourier Transform (FFT) on the raw ADC data.

    Args:
        data_frame (pandas.DataFrame): The input data containing the raw ADC values.
        sampling_rate (int): The sampling rate of the sensor.

    Returns:
        tuple: A tuple containing the frequencies and the corresponding FFT magnitudes.
               Returns (None, None) if the ADC data is not found.
    """
    # The raw ADC data starts from column index ADC_DATA_START_INDEX
    adc_data = data_frame.iloc[:, ADC_DATA_START_INDEX:].values.flatten()
    
    if adc_data.size == 0:
        print("Error: No ADC data found from the specified start index.")
        return None, None

    # Perform FFT
    fft_result = np.fft.fft(adc_data)
    fft_magnitude = np.abs(fft_result)
    
    # Generate the frequency axis
    n = len(adc_data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # We are only interested in the positive frequencies
    positive_frequencies = frequencies[:n//2]
    positive_magnitudes = fft_magnitude[:n//2]
    
    return positive_frequencies, positive_magnitudes

def extract_spectral_features(frequencies, magnitudes):
    """
    Extract spectral features from the FFT result.

    Args:
        frequencies (numpy.ndarray): The array of frequencies from the FFT.
        magnitudes (numpy.ndarray): The array of corresponding magnitudes from the FFT.

    Returns:
        dict: A dictionary containing the extracted spectral features:
              - Peak Frequency
              - Mean Frequency
              - Spectral Centroid
              - Spectral Skewness
              - Spectral Kurtosis
              Returns None if the input arrays are empty or invalid.
    """
    if frequencies is None or magnitudes is None or frequencies.size == 0 or magnitudes.size == 0:
        return None

    # 1. Peak Frequency: The frequency with the highest magnitude
    peak_frequency = frequencies[np.argmax(magnitudes)]
    
    # 2. Mean Frequency: The weighted average of the frequencies
    mean_frequency = np.average(frequencies, weights=magnitudes)
    
    # 3. Spectral Centroid: The center of mass of the spectrum
    # This is similar to the mean frequency but is a standard feature in audio and signal processing.
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)

    # 4. Spectral Skewness: Measure of the asymmetry of the spectral distribution
    spectral_skewness = skew(magnitudes)

    # 5. Spectral Kurtosis: Measure of the "tailedness" of the spectral distribution
    # High kurtosis indicates more outliers (sharp peaks).
    spectral_kurtosis = np.sum(((frequencies - mean_frequency)**4) * magnitudes) / (np.sum(magnitudes) * (np.std(frequencies)**4))


    return {
        "Peak Frequency": peak_frequency,
        "Mean Frequency": mean_frequency,
        "Spectral Centroid": spectral_centroid,
        "Spectral Skewness": spectral_skewness,
        "Spectral Kurtosis": spectral_kurtosis
    }

def main():
    """
    Main function to demonstrate the data processing pipeline.
    """
    # --- Parameters ---
    # The sampling rate is the number of samples taken per second.
    SAMPLING_RATE = 1953125

    # Choose one of your data files to process
    # Note: We are reading only the first 10 rows for demonstration purposes, as requested.
    # For actual processing, you would typically process a single complete measurement.
    data_file = 'data/raw/signal_1500_metal_plate.csv'
    
    # --- Processing ---
    print(f"Loading data from '{data_file}'...")
    df = load_data(data_file)
    
    if df is not None:
        print("Performing FFT...")
        frequencies, magnitudes = perform_fft(df, SAMPLING_RATE)
        
        if frequencies is not None:
            print("Extracting spectral features...")
            features = extract_spectral_features(frequencies, magnitudes)
            
            if features:
                print("""
--- Extracted Features ---""")
                for feature, value in features.items():
                    print(f"{feature}: {value:.2f}")
                print("""--------------------------
""")
                
                # --- Interpretation ---
                print("--- Interpretation ---")
                print(f"The sensor's emitted frequency is {SENSOR_FREQUENCY / 1000} kHz.")
                print("If an object is moving towards the sensor, we expect the 'Peak Frequency' and 'Mean Frequency' to be slightly higher than this.")
                print("If it's moving away, they would be lower.")
                print("These features can now be used to train a classifier.")
                print("----------------------")


if __name__ == "__main__":
    main()
