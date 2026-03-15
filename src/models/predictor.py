"""
Real-time Prediction Simulation: Uses PURE SENSOR DATA to detect approaches.
Implements QUAD-SCALE temporal windows (5, 10, 25, 50) for dynamic inference.

FIX (B1) — Shared echo constants:
    ECHO_MIN_INDEX and ECHO_THRESHOLD_MULTIPLIER are now imported from
    processing.py, matching the values used in feature_extractor.py exactly.

FIX (B2) — Per-frame Kalman dt:
    The Kalman filter is initialised with the mean dt from the file's timestamps,
    but each update() call now also passes the real inter-frame interval. This
    keeps kinematics synchronised with the actual sensor clock even when the
    frame rate jitters.

FIX (D) — sklearn Pipeline:
    The predictor loads a single Pipeline artifact (StandardScaler + model).
    There is no separate scaler file. Calling pipeline.predict(X_raw) applies
    scaling automatically in the correct column order — making feature-ordering
    bugs impossible at inference time.
"""
import os
import sys
import time
import pandas as pd
import joblib
import argparse
import numpy as np
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.processing import (
    ADC_DATA_START_INDEX,
    ECHO_MIN_INDEX,               # FIX (B1): shared constant
    ECHO_THRESHOLD_MULTIPLIER,    # FIX (B1): shared constant
    find_first_peak_index,
    extract_echo_shape_features,
)
from src.data.kalman import KalmanFilter
from src.models.detectors import AutoencoderDetector

# Quad-scale windows — must match model_trainer.py and feature_extractor.py
WINDOWS = [5, 10, 25, 50]


def _build_motion_feature_vector(echo_idx, echo_shape, peak_freq, spectral_centroid,
                                  buffer_echo, buffer_centroid):
    """
    Construct the ordered 17-feature dict used by the motion Pipeline.

    FIX (D): column order is derived from the same helper used in model_trainer.py.
    The Pipeline's internal StandardScaler handles scaling; we pass raw values.
    """
    feat = {
        'echo_index':        echo_idx,
        'echo_amplitude':    echo_shape['echo_amplitude'],
        'echo_width':        echo_shape['echo_width'],
        'Peak Frequency':    peak_freq,
        'Spectral Centroid': spectral_centroid,
    }
    for w in WINDOWS:
        feat[f'Trend_{w}']         = buffer_echo[-1]     - buffer_echo[-w]
        feat[f'Mean_{w}']          = np.mean(list(buffer_echo)[-w:])
        feat[f'Centroid_Trend_{w}'] = buffer_centroid[-1] - buffer_centroid[-w]
    return feat


def main():
    parser = argparse.ArgumentParser(description='Simulate real-time tracking on raw sensor data.')
    parser.add_argument('filepath', type=str, help='Path to a raw sensor CSV file.')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load Pipeline artifacts  (FIX D: single file, no separate scaler)
    # ------------------------------------------------------------------
    try:
        motion_pipeline = joblib.load('models/motion_detection_model.joblib')
        ttc_pipeline    = joblib.load('models/ttc_prediction_model.joblib')
        logger.info("Loaded model pipelines successfully.")
    except Exception as e:
        logger.error(f"Could not load pipelines: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Batch pre-computation (FFT over the whole file)
    # ------------------------------------------------------------------
    logger.info(f"Analysing: {os.path.basename(args.filepath)}")
    df_raw        = pd.read_csv(args.filepath, header=None)
    SAMPLING_RATE = 1953125

    adc_matrix   = df_raw.iloc[:, ADC_DATA_START_INDEX:].values
    adc_centered = adc_matrix - np.mean(adc_matrix, axis=1, keepdims=True)

    fft_results = np.fft.fft(adc_centered, axis=1)
    magnitudes  = np.abs(fft_results[:, 1:adc_matrix.shape[1] // 2])
    frequencies = np.fft.fftfreq(
        adc_matrix.shape[1], d=1 / SAMPLING_RATE
    )[1:adc_matrix.shape[1] // 2]

    peak_freqs         = frequencies[np.argmax(magnitudes, axis=1)]
    sum_mags           = np.sum(magnitudes, axis=1)
    spectral_centroids = np.sum(frequencies * magnitudes, axis=1) / sum_mags

    # ------------------------------------------------------------------
    # 3. Kalman Filter initialisation  (FIX B2: mean dt from timestamps)
    # ------------------------------------------------------------------
    timestamps_ms = df_raw.iloc[:, 16].values
    dt_avg = np.mean(np.diff(timestamps_ms)) / 1000.0
    if np.isnan(dt_avg) or dt_avg <= 0:
        dt_avg = 0.045
    kf = KalmanFilter(dt=dt_avg)
    logger.info(f"Kalman filter dt initialised from timestamps: {dt_avg*1000:.2f} ms")

    # ------------------------------------------------------------------
    # 4. Streaming simulation — one row at a time
    # ------------------------------------------------------------------
    MAX_W           = max(WINDOWS)
    buffer_echo     = deque(maxlen=MAX_W)
    buffer_centroid = deque(maxlen=MAX_W)

    towards_count   = 0
    predicted_ttcs  = []
    frame_latencies = []

    # Column order must match _build_classification_features() in model_trainer.py
    motion_cols = ['echo_index', 'echo_amplitude', 'echo_width',
                   'Peak Frequency', 'Spectral Centroid']
    for w in WINDOWS:
        motion_cols.extend([f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}'])

    ttc_cols = [
        'velocity', 'acceleration', 'echo_index',
        'Peak Frequency', 'Spectral Centroid', 'Trend_25',
    ]

    for i in range(len(df_raw)):

        # FIX (B2): compute actual inter-frame dt and pass it to the Kalman update
        if i > 0:
            dt_frame = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
            if dt_frame <= 0 or np.isnan(dt_frame):
                dt_frame = None          # fallback: KalmanFilter uses constructor dt
        else:
            dt_frame = None

        _, f_vel, f_accel = kf.update(df_raw.iloc[i, 10], dt=dt_frame)

        # --- Echo detection (FIX B1: shared constants from processing.py) ---
        echo_idx = float(
            find_first_peak_index(
                adc_centered[i],
                threshold_multiplier=ECHO_THRESHOLD_MULTIPLIER,
                min_index=ECHO_MIN_INDEX,
            )
        )

        # --- Echo shape features ---
        echo_shape = extract_echo_shape_features(
            adc_centered[i], int(echo_idx) if echo_idx >= 0 else -1
        )

        buffer_echo.append(echo_idx)
        buffer_centroid.append(spectral_centroids[i])

        if len(buffer_echo) < MAX_W:
            continue

        # --- Start latency clock (ML prediction only) ---
        t_start = time.perf_counter()

        # --- Build feature vector and predict (FIX D: Pipeline handles scaling) ---
        feat_dict = _build_motion_feature_vector(
            echo_idx, echo_shape, peak_freqs[i], spectral_centroids[i],
            buffer_echo, buffer_centroid,
        )
        X_motion = pd.DataFrame([feat_dict])[motion_cols]

        # FIX (D): pipeline.predict() applies StandardScaler then model internally
        prediction = motion_pipeline.predict(X_motion)[0]

        if prediction == 1:
            towards_count += 1

            ttc_feat = {
                'velocity':          f_vel,
                'acceleration':      f_accel,
                'echo_index':        echo_idx,
                'Peak Frequency':    peak_freqs[i],
                'Spectral Centroid': spectral_centroids[i],
                'Trend_25':          buffer_echo[-1] - list(buffer_echo)[-25],
            }
            X_ttc = pd.DataFrame([ttc_feat])[ttc_cols]

            # FIX (D): ttc_pipeline handles its own scaling
            predicted_ttcs.append(max(0.0, ttc_pipeline.predict(X_ttc)[0]))

        # --- Stop latency clock ---
        t_end = time.perf_counter()
        frame_latencies.append((t_end - t_start) * 1000.0)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    n_frames        = len(frame_latencies)
    mean_latency_ms = np.mean(frame_latencies)  if n_frames else 0.0
    max_latency_ms  = np.max(frame_latencies)   if n_frames else 0.0
    p99_latency_ms  = np.percentile(frame_latencies, 99) if n_frames else 0.0
    approx_period_ms = (5 * 60 * 1000) / max(len(df_raw), 1)

    logger.info("Analysis complete.")
    print("\n" + "=" * 45)
    print("QUAD-SCALE DETECTION REPORT")
    print("=" * 45)
    print(f"File           : {os.path.basename(args.filepath)}")
    print(f"Frames analysed: {len(df_raw)}")
    print(f"Towards frames : {towards_count}")

    if predicted_ttcs:
        print(f"Min TTC        : {min(predicted_ttcs):.2f} s")
        print(f"Mean TTC       : {np.mean(predicted_ttcs):.2f} s")

    print(f"Movement present: {'YES' if towards_count > 15 else 'NO'}")

    print("\n--- Latency (ML prediction only) ---")
    print(f"  Mean          : {mean_latency_ms:.3f} ms")
    print(f"  Max           : {max_latency_ms:.3f} ms")
    print(f"  p99           : {p99_latency_ms:.3f} ms")
    print(f"  Sensor period : ~{approx_period_ms:.1f} ms/frame")

    if mean_latency_ms < approx_period_ms:
        print(f"  Verdict       : REAL-TIME CAPABLE "
              f"(mean {mean_latency_ms:.2f} ms < {approx_period_ms:.1f} ms period)")
    else:
        print("  Verdict       : WARNING — mean prediction latency exceeds sensor period.")

    print("=" * 45)


if __name__ == "__main__":
    main()
