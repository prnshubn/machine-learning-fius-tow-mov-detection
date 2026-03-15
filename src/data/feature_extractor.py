"""
Feature Engineering Module: Extracts raw sensor features from radar ADC data.
Uses QUAD-SCALE TEMPORAL WINDOWS (5, 10, 25, 50) for robust pattern recognition.

FIX (B1) — Shared echo constants:
    ECHO_MIN_INDEX and ECHO_THRESHOLD_MULTIPLIER are now imported from
    processing.py instead of being hardcoded. This guarantees that the same
    blanking window and sigma multiplier are used during training (here) AND
    during inference (predictor.py), so echo_index distributions match exactly.
"""
import os
import glob
import sys
import pandas as pd
import numpy as np
import logging
from src.data.processing import (
    ADC_DATA_START_INDEX,
    ECHO_MIN_INDEX,
    ECHO_THRESHOLD_MULTIPLIER,
    find_first_peak_index,
    extract_echo_shape_features,
)
from src.data.kalman import apply_kalman_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_feature_dataset():
    """
    Extract features using Short (5, 10), Mid (25), and Long (50) temporal windows.

    Feature groups produced per row:
      Kinematics  : velocity, acceleration  (from Kalman filter)
      Echo timing : echo_index              (first peak arrival sample)
      Echo shape  : echo_amplitude, echo_width (amplitude and FWHM)
      Spectral    : Peak Frequency, Spectral Centroid
      Quad-scale  : Trend_W, Mean_W, Centroid_Trend_W  for W in {5, 10, 25, 50}
    """
    SAMPLING_RATE = 1953125
    WINDOWS       = [5, 10, 25, 50]
    raw_data_path       = 'data/raw'
    processed_data_path = 'data/processed'
    output_filepath     = os.path.join(processed_data_path, 'features.csv')

    os.makedirs(processed_data_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))

    if not csv_files:
        logger.error("No raw data files found in data/raw/")
        sys.exit(1)

    logger.info(
        f"Starting Quad-Scale extraction with windows {WINDOWS}, "
        f"echo min_index={ECHO_MIN_INDEX}, "
        f"threshold_multiplier={ECHO_THRESHOLD_MULTIPLIER}..."
    )
    all_features_dfs = []

    for filepath in csv_files:
        filename     = os.path.basename(filepath)
        object_label = "_".join(filename.replace('.csv', '').split('_')[2:])

        try:
            df         = pd.read_csv(filepath, header=None)
            adc_matrix = df.iloc[:, ADC_DATA_START_INDEX:].values
            adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)

            # ------------------------------------------------------------------
            # 1. Echo Detection — uses shared constants from processing.py
            #    FIX (B1): min_index=ECHO_MIN_INDEX matches predictor.py exactly.
            #    FIX (C):  find_first_peak_index now computes std only on the
            #              post-blanking region, so weak echoes are not masked.
            # ------------------------------------------------------------------
            echo_indices = np.array([
                find_first_peak_index(
                    row,
                    threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER,
                    min_index=ECHO_MIN_INDEX,
                )
                for row in adc_centered
            ], dtype=float)

            # ------------------------------------------------------------------
            # 2. Echo Shape Features
            # ------------------------------------------------------------------
            echo_shapes     = [
                extract_echo_shape_features(row, int(idx) if idx >= 0 else -1)
                for row, idx in zip(adc_centered, echo_indices)
            ]
            echo_amplitudes = np.array([s['echo_amplitude'] for s in echo_shapes])
            echo_widths     = np.array([s['echo_width']     for s in echo_shapes])

            # ------------------------------------------------------------------
            # 3. Spectral Features
            # ------------------------------------------------------------------
            fft_results  = np.fft.fft(adc_centered, axis=1)
            n_samples    = adc_matrix.shape[1]
            magnitudes   = np.abs(fft_results[:, 1:n_samples // 2])
            frequencies  = np.fft.fftfreq(n_samples, d=1 / SAMPLING_RATE)[1:n_samples // 2]
            peak_freqs        = frequencies[np.argmax(magnitudes, axis=1)]
            spectral_centroids = (
                np.sum(frequencies * magnitudes, axis=1)
                / np.sum(magnitudes, axis=1)
            )

            # ------------------------------------------------------------------
            # 4. Kinematics via Kalman Filter
            #    Column 10 = distance measurement, column 16 = timestamp (ms)
            # ------------------------------------------------------------------
            _, velocity, acceleration = apply_kalman_filter(
                df.iloc[:, 10], df.iloc[:, 16]
            )

            # ------------------------------------------------------------------
            # 5. Quad-Scale Temporal Features
            # ------------------------------------------------------------------
            temp_df = pd.DataFrame({
                'echo':     echo_indices,
                'centroid': spectral_centroids,
            })

            feature_dict = {
                'label':             object_label,
                'timestamp':         df.iloc[:, 16].values,
                'velocity':          velocity,
                'acceleration':      acceleration,
                'echo_index':        echo_indices,
                'echo_amplitude':    echo_amplitudes,
                'echo_width':        echo_widths,
                'Peak Frequency':    peak_freqs,
                'Spectral Centroid': spectral_centroids,
            }

            for w in WINDOWS:
                feature_dict[f'Trend_{w}']         = temp_df['echo'].diff(periods=w - 1).fillna(0).values
                feature_dict[f'Mean_{w}']           = temp_df['echo'].rolling(window=w, min_periods=1).mean().values
                feature_dict[f'Centroid_Trend_{w}'] = temp_df['centroid'].diff(periods=w - 1).fillna(0).values

            all_features_dfs.append(pd.DataFrame(feature_dict))
            logger.info(f"  Processed: {filename} ({len(df)} rows, label='{object_label}')")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    if not all_features_dfs:
        logger.error("No feature data was produced. Check raw files and ADC column layout.")
        sys.exit(1)

    final_dataset = pd.concat(all_features_dfs, ignore_index=True)
    final_dataset.to_csv(output_filepath, index=False)
    logger.info(
        f"Feature dataset saved to {output_filepath} "
        f"({len(final_dataset)} rows, {len(final_dataset.columns)} columns)"
    )


if __name__ == "__main__":
    build_feature_dataset()
