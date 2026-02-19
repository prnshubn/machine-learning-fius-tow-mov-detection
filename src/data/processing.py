
"""
This script provides a framework for processing ultrasonic sensor data to detect moving objects.
It includes functions to load data, perform a Fast Fourier Transform (FFT) to analyze the frequency spectrum,
and extract relevant features from the spectrum. These features can then be used to train a machine learning model.

Main functionalities:
- Load sensor data from CSV files
- Perform FFT to extract frequency domain features
- Calculate kinematic features (velocity, acceleration)
- Utility functions for feature extraction and peak detection
"""


# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
from scipy.stats import skew  # For calculating skewness


# --- Constants ---
SENSOR_FREQUENCY = 40000  # 40 kHz, the frequency of the emitted ultrasonic pulse
ADC_DATA_START_INDEX = 17 # The starting index of the raw ADC data in the CSV file


def load_data(filepath, nrows=None):
    """
    Load data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        nrows (int, optional): The number of rows to read. Defaults to None (read all rows).

    Returns:
        pandas.DataFrame: The loaded data, or None if file not found or error occurs.
    """
    try:
        # Attempt to read the CSV file
        return pd.read_csv(filepath, nrows=nrows)
    except FileNotFoundError:
        # Handle missing file
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        # Handle other errors
        print(f"An error occurred while reading the file: {e}")
        return None


def perform_fft(data_frame, sampling_rate):
    """
    Perform a Fast Fourier Transform (FFT) on the raw ADC data.
    Removes the DC offset (mean) to ensure peak frequency detection is accurate.

    Args:
        data_frame (pandas.DataFrame): The input data containing the raw ADC values.
        sampling_rate (int): The sampling rate of the sensor.

    Returns:
        tuple: (frequencies, magnitudes) or (None, None) if ADC data is missing.
    """
    # Extract raw ADC data starting from ADC_DATA_START_INDEX
    adc_data = data_frame.iloc[:, ADC_DATA_START_INDEX:].values.flatten()
    
    if adc_data.size == 0:
        # No ADC data found
        print("Error: No ADC data found from the specified start index.")
        return None, None

    # Remove DC Offset by subtracting the mean
    adc_data_centered = adc_data - np.mean(adc_data)

    # Perform FFT on the centered data
    fft_result = np.fft.fft(adc_data_centered)
    # ...existing code...
    fft_magnitude = np.abs(fft_result)
    
    # Generate the frequency axis
    n = len(adc_data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # We are only interested in the positive frequencies.
    # We skip the first bin (index 0) which is the DC component (now ~0) 
    # to avoid it being picked as the 'Peak Frequency'.
    positive_frequencies = frequencies[1:n//2]
    positive_magnitudes = fft_magnitude[1:n//2]
    
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

def find_first_peak_index(adc_data, threshold_multiplier=3):
    """
    Finds the index of the first peak in the ADC data that exceeds a threshold.
    The threshold is calculated as mean + (threshold_multiplier * std).
    
    Args:
        adc_data (numpy.ndarray): The array of ADC values.
        threshold_multiplier (float): Multiplier for the standard deviation to set the threshold.
        
    Returns:
        int: The index of the first peak. Returns -1 if no peak is found.
    """
    if adc_data is None or adc_data.size == 0:
        return -1
        
    mean_val = np.mean(adc_data)
    std_val = np.std(adc_data)
    threshold = mean_val + (threshold_multiplier * std_val)
    
    # Find indices where value > threshold
    peaks = np.where(adc_data > threshold)[0]
    
    if peaks.size > 0:
        return peaks[0]
    else:
        return -1

def calculate_kinematics(distances, timestamps):
    """
    Calculates velocity and acceleration from distance and timestamp arrays.
    
    Args:
        distances (pandas.Series or numpy.ndarray): Array of distance measurements.
        timestamps (pandas.Series or numpy.ndarray): Array of timestamps (in milliseconds).
        
    Returns:
        tuple: (velocity, acceleration) as pandas.Series or numpy.ndarray.
    """
    # Ensure inputs are pandas Series for easy shifting
    if not isinstance(distances, pd.Series):
        distances = pd.Series(distances)
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
        
    # Calculate delta distance and delta time
    delta_dist = distances.diff()
    delta_time = timestamps.diff()
    
    # Avoid division by zero
    delta_time = delta_time.replace(0, np.nan)
    
    # Calculate Velocity (distance / time)
    # Assuming distance units and time units (e.g., cm/ms or m/s)
    velocity = delta_dist / delta_time
    
    # Calculate Acceleration (velocity / time)
    delta_velocity = velocity.diff()
    acceleration = delta_velocity / delta_time
    
    # Fill NaN values resulting from diff() with 0 or the first valid value
    velocity = velocity.fillna(0)
    acceleration = acceleration.fillna(0)
    
    return velocity, acceleration

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
