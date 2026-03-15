"""
This script provides a framework for processing ultrasonic sensor data to detect moving objects.
It includes functions to load data, perform a Fast Fourier Transform (FFT) to analyze the frequency
spectrum, and extract relevant features from the spectrum. These features can then be used to train
a machine learning model.

Main functionalities:
- Load sensor data from CSV files
- Perform FFT to extract frequency domain features
- Calculate kinematic features (velocity, acceleration)
- Utility functions for feature extraction and peak detection

FIX (C) — Noise-floor std: find_first_peak_index now computes std only on
    the post-blanking region (samples > min_index), so the high-amplitude
    transmitted pulse no longer inflates the detection threshold. This makes
    the detector more sensitive to weak echoes from soft or distant objects.

FIX (B1) — Shared constants: ECHO_MIN_INDEX and ECHO_THRESHOLD_MULTIPLIER
    are now defined here and imported by both feature_extractor.py and
    predictor.py, guaranteeing identical echo_index distributions at train
    and inference time.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew


# --- Constants ---
SENSOR_FREQUENCY = 40000   # 40 kHz — frequency of the emitted ultrasonic pulse
ADC_DATA_START_INDEX = 17  # Starting column index of the raw ADC data in CSV files

# Shared echo-detection parameters — imported by feature_extractor.py AND predictor.py.
# Changing a value here propagates to both train and inference automatically.
ECHO_MIN_INDEX            = 20  # Blank first 20 samples (transmitted-pulse zone)
ECHO_THRESHOLD_MULTIPLIER = 4   # 4-sigma detection threshold


def load_data(filepath, nrows=None):
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Defaults to None (read all).

    Returns:
        pandas.DataFrame | None
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
    Removes the DC offset (mean) to ensure peak frequency detection is accurate.

    Args:
        data_frame (pandas.DataFrame): Input data containing raw ADC values.
        sampling_rate (int): Sampling rate of the sensor in Hz.

    Returns:
        tuple: (frequencies, magnitudes) or (None, None) if ADC data is missing.
    """
    adc_data = data_frame.iloc[:, ADC_DATA_START_INDEX:].values.flatten()

    if adc_data.size == 0:
        print("Error: No ADC data found from the specified start index.")
        return None, None

    # Remove DC offset
    adc_data_centered = adc_data - np.mean(adc_data)

    fft_result = np.fft.fft(adc_data_centered)
    fft_magnitude = np.abs(fft_result)

    n = len(adc_data)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    positive_frequencies = frequencies[1:n // 2]
    positive_magnitudes  = fft_magnitude[1:n // 2]

    return positive_frequencies, positive_magnitudes


def extract_spectral_features(frequencies, magnitudes):
    """
    Extract spectral features from the FFT result.

    Args:
        frequencies (numpy.ndarray): Frequency axis from the FFT.
        magnitudes  (numpy.ndarray): Corresponding magnitude values.

    Returns:
        dict | None
    """
    if frequencies is None or magnitudes is None or frequencies.size == 0 or magnitudes.size == 0:
        return None

    peak_frequency    = frequencies[np.argmax(magnitudes)]
    mean_frequency    = np.average(frequencies, weights=magnitudes)
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    spectral_skewness = skew(magnitudes)
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
    Find the index of the first echo peak in DC-centred ADC data.

    Uses a two-sided absolute-value threshold so that both positive and negative
    reflection peaks are detected.

    FIX (C) — Noise-floor estimation:
        Previously, std was computed over the ENTIRE window, meaning the large
        transmitted pulse at the start inflated std and raised the detection
        threshold — making the detector blind to weak echoes from soft/distant
        objects (e.g. humans at range).

        Now, std is computed only on the post-blanking region (samples after
        min_index). This estimates the true noise floor without contamination
        from the TX pulse, making the detector significantly more sensitive.

    Args:
        adc_data           (numpy.ndarray): DC-centred ADC row.
        threshold_multiplier (float): Sigma multiplier for the detection threshold.
            Default 3 (3-sigma). Use ECHO_THRESHOLD_MULTIPLIER (4) for training.
        min_index (int): Ignore any peaks at sample indices <= this value.
            Blanks the transmitted pulse zone. Use ECHO_MIN_INDEX (20) everywhere.

    Returns:
        int: Index of the first detected echo peak, or -1 if none found.
    """
    if adc_data is None or adc_data.size == 0:
        return -1

    # FIX (C): estimate noise floor only on the blanked (post-TX-pulse) region
    if min_index > 0 and min_index < len(adc_data):
        noise_region = adc_data[min_index:]
    else:
        noise_region = adc_data

    std_val = np.std(noise_region)
    if std_val == 0:
        return -1

    threshold = threshold_multiplier * std_val

    # Two-sided: detect positive AND negative reflection peaks
    peaks = np.where(np.abs(adc_data) > threshold)[0]

    # Blank the transmitted-pulse zone
    if min_index > 0:
        peaks = peaks[peaks > min_index]

    return int(peaks[0]) if peaks.size > 0 else -1


def extract_echo_shape_features(adc_data, echo_index):
    """
    Extract amplitude and Full-Width at Half-Maximum (FWHM) width of a detected echo.

    Args:
        adc_data    (numpy.ndarray): DC-centred ADC row.
        echo_index  (int): Sample index of the detected peak. Pass -1 for no detection.

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

    left = echo_index
    while left > 0 and np.abs(adc_data[left - 1]) >= half_max:
        left -= 1

    right = echo_index
    while right < n - 1 and np.abs(adc_data[right + 1]) >= half_max:
        right += 1

    width = float(right - left + 1)

    return {'echo_amplitude': amplitude, 'echo_width': width}


def calculate_kinematics(distances, timestamps):
    """
    Calculate velocity and acceleration from distance and timestamp arrays.

    Args:
        distances  (pandas.Series | numpy.ndarray): Distance measurements.
        timestamps (pandas.Series | numpy.ndarray): Timestamps in milliseconds.

    Returns:
        tuple: (velocity, acceleration) as pandas.Series.
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
    """Demonstrate the processing pipeline on a sample file."""
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
                print("Towards → Peak/Mean Frequency slightly above emitted frequency (Doppler).")
                print("Away    → Peak/Mean Frequency slightly below emitted frequency.")
                print("----------------------")


if __name__ == "__main__":
    main()
