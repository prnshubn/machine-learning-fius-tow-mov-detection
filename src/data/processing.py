"""
Signal Processing Module: Utilities for Ultrasonic Data Analysis.

This module provides the core signal processing algorithms for the project,
including Fast Fourier Transform (FFT) analysis, spectral feature extraction,
and noise-adaptive echo peak detection.

Key Components:
- FFT Analysis: Converts time-domain ADC samples into frequency-domain magnitudes.
- Spectral Features: Calculates centroid, skewness, and kurtosis to characterize echoes.
- Peak Detection: Implements a 3-sigma noise-floor-adaptive search for the first echo.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew


# --- Constants ---
SENSOR_FREQUENCY = 40000   # 40 kHz center frequency of the ultrasonic transducer
ADC_DATA_START_INDEX = 17  # Index where raw ADC voltage samples begin in the CSV

# Shared parameters for peak detection consistency
# These are used by both training (feature_extractor) and inference (predictor)
ECHO_MIN_INDEX            = 20  # Skip first 20 samples to avoid Tx pulse contamination
ECHO_THRESHOLD_MULTIPLIER = 4   # Use 4-sigma for robust noise rejection


def load_data(filepath, nrows=None):
    """
    Loads raw sensor data from a CSV file.

    Args:
        filepath (str): Path to the CSV.
        nrows (int, optional): Number of rows for partial reading.

    Returns:
        pd.DataFrame or None: Loaded data or None if file error occurs.
    """
    try:
        return pd.read_csv(filepath, nrows=nrows)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def perform_fft(data_frame, sampling_rate):
    """
    Computes the Fast Fourier Transform of the ADC signal.

    The signal is first DC-centered (mean removed) to prevent the DC bin (0 Hz)
    from dominating the spectral analysis.

    Args:
        data_frame (pd.DataFrame): Data containing ADC columns.
        sampling_rate (int): Sampling frequency in Hz.

    Returns:
        tuple: (positive_frequencies, positive_magnitudes)
    """
    adc_data = data_frame.iloc[:, ADC_DATA_START_INDEX:].values.flatten()

    if adc_data.size == 0:
        return None, None

    # DC-offset removal (centering around zero)
    adc_data_centered = adc_data - np.mean(adc_data)

    fft_result = np.fft.fft(adc_data_centered)
    fft_magnitude = np.abs(fft_result)

    n = len(adc_data)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Keep only the positive half of the spectrum (symmetry)
    positive_frequencies = frequencies[1:n // 2]
    positive_magnitudes  = fft_magnitude[1:n // 2]

    return positive_frequencies, positive_magnitudes


def extract_spectral_features(frequencies, magnitudes):
    """
    Extracts statistical descriptors from the frequency spectrum.

    Args:
        frequencies (np.ndarray): Spectral frequency axis.
        magnitudes (np.ndarray): Magnitude values.

    Returns:
        dict: Features including Peak Freq, Centroid, Skewness, and Kurtosis.
    """
    if frequencies is None or magnitudes is None or frequencies.size == 0:
        return None

    peak_frequency    = frequencies[np.argmax(magnitudes)]
    mean_frequency    = np.average(frequencies, weights=magnitudes)
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    spectral_skewness = skew(magnitudes)
    
    # Spectral Kurtosis: measures the 'peakedness' of the spectrum
    spectral_kurtosis = (
        np.sum(((frequencies - mean_frequency) ** 4) * magnitudes)
        / (np.sum(magnitudes) * (np.std(frequencies) ** 4))
    )

    return {
        "Peak Frequency":    peak_frequency,
        "Mean Frequency":    mean_frequency,
        "Spectral Centroid": spectral_centroid,
        "Spectral Skewness": spectral_skewness,
        "Spectral Kurtosis": spectral_kurtosis,
    }


def find_first_peak_index(adc_data, threshold_multiplier=3, min_index=0):
    """
    Detects the first significant echo peak using an adaptive threshold.

    The threshold is defined as (multiplier * noise_std). To ensure sensitivity
    to distant objects, the noise floor (std) is estimated ONLY on the samples
    after the initial blanking period (min_index).

    Args:
        adc_data (np.ndarray): DC-centered ADC signal row.
        threshold_multiplier (float): Sensitivity factor (sigma multiplier).
        min_index (int): Number of initial samples to ignore (Tx pulse zone).

    Returns:
        int: Sample index of the first peak, or -1 if not found.
    """
    if adc_data is None or adc_data.size == 0:
        return -1

    # Estimate noise floor on the valid echo region only (after Tx pulse)
    if min_index > 0 and min_index < len(adc_data):
        noise_region = adc_data[min_index:]
    else:
        noise_region = adc_data

    std_val = np.std(noise_region)
    if std_val == 0:
        return -1

    threshold = threshold_multiplier * std_val

    # Detect all points crossing the threshold (two-sided for phase robustness)
    peaks = np.where(np.abs(adc_data) > threshold)[0]

    # Filter out samples in the blanking zone
    if min_index > 0:
        peaks = peaks[peaks > min_index]

    return int(peaks[0]) if peaks.size > 0 else -1


def extract_echo_shape_features(adc_data, echo_index):
    """
    Computes the amplitude and FWHM (Full-Width at Half-Maximum) of a peak.

    Width helps distinguish surface materials (hard surfaces return sharp/narrow
    echoes, soft surfaces return broad/diffuse ones).

    Args:
        adc_data (np.ndarray): ADC signal row.
        echo_index (int): Index of the detected peak.

    Returns:
        dict: {'echo_amplitude': float, 'echo_width': float}
    """
    n = len(adc_data)

    if echo_index < 0 or echo_index >= n:
        return {'echo_amplitude': 0.0, 'echo_width': 0.0}

    amplitude = float(np.abs(adc_data[echo_index]))
    if amplitude == 0.0:
        return {'echo_amplitude': 0.0, 'echo_width': 0.0}

    half_max = amplitude / 2.0

    # Walk left to find half-max boundary
    left = echo_index
    while left > 0 and np.abs(adc_data[left - 1]) >= half_max:
        left -= 1

    # Walk right to find half-max boundary
    right = echo_index
    while right < n - 1 and np.abs(adc_data[right + 1]) >= half_max:
        right += 1

    width = float(right - left + 1)
    return {'echo_amplitude': amplitude, 'echo_width': width}


def calculate_kinematics(distances, timestamps):
    """
    Calculates raw velocity and acceleration (simple finite difference).

    Note: For production inference, use the Kalman Filter instead of this function.

    Args:
        distances (pd.Series): Distance measurements.
        timestamps (pd.Series): Timestamps in milliseconds.

    Returns:
        tuple: (velocity, acceleration)
    """
    if not isinstance(distances, pd.Series):
        distances = pd.Series(distances)
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    delta_dist = distances.diff()
    delta_time = timestamps.diff().replace(0, np.nan)

    velocity     = delta_dist / delta_time
    acceleration = velocity.diff() / delta_time

    return velocity.fillna(0), acceleration.fillna(0)


def main():
    """Demonstrates the processing pipeline on a sample file."""
    SAMPLING_RATE = 1953125
    data_file = 'data/raw/signal_1500_metal_plate.csv'

    print(f"Loading data from '{data_file}'...")
    df = load_data(data_file)

    if df is not None:
        print("Performing FFT...")
        frequencies, magnitudes = perform_fft(df, SAMPLING_RATE)

        if frequencies is not None:
            print("Extracting spectral features...")
            features = extract_spectral_features(frequencies, magnitudes)

            if features:
                print("\n--- Extracted Features ---")
                for feature, value in features.items():
                    print(f"  {feature}: {value:.2f}")
                print("--------------------------\n")
                print("--- Interpretation ---")
                print(f"Emitted frequency: {SENSOR_FREQUENCY / 1000} kHz")
                print("Towards → Peak/Mean Frequency slightly above center (Doppler).")
                print("Away    → Peak/Mean Frequency slightly below center.")
                print("----------------------")


if __name__ == "__main__":
    main()
