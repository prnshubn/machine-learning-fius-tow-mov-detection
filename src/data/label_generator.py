"""
Label Refinement Module: Generates binary ground truth labels (towards vs not_towards).
Captures multiple movement bursts within a single session using dynamic thresholding.

FIX (A) — Kinematic TTC (quadratic):
    The original ground truth used TTC = distance / |velocity|, which assumes
    constant velocity. An accelerating hand invalidates that assumption.

    The Kalman filter already tracks acceleration, so we can solve the full
    kinematic equation for impact time:

        d + v*t + 0.5*a*t² = 0

    Rearranged: 0.5*a*t² + v*t + d = 0

    This is a standard quadratic in t. We take the smallest positive real root.
    When acceleration is negligible (|a| < 1e-6) we fall back to the linear
    formula to avoid division-by-zero. The result is a higher-quality training
    signal for the SVR, particularly during fast or jerky approach profiles.

Other improvements (unchanged from previous version):
  - Boundary erosion: transition frames stripped from each end of 'towards' blocks.
  - Majority-vote smoothing (window=3) to remove single-frame noise.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOUNDARY_EROSION_FRAMES = 2


# ---------------------------------------------------------------------------
# FIX (A) — Kinematic TTC helper
# ---------------------------------------------------------------------------

def _kinematic_ttc(distance, velocity, acceleration):
    """
    Solve d + v*t + 0.5*a*t² = 0 for the smallest positive real root.

    This accounts for constant acceleration and provides a more accurate
    ground-truth TTC than the constant-velocity approximation.

    Args:
        distance     (float): Current sensor-to-object distance in metres (> 0).
        velocity     (float): Kalman-filtered velocity in m/s (negative = approaching).
        acceleration (float): Kalman-filtered acceleration in m/s².

    Returns:
        float: Predicted time to contact in seconds, or np.nan if no valid root.
    """
    if distance <= 0 or np.isnan(distance):
        return np.nan

    if abs(acceleration) < 1e-6:
        # Linear fallback: t = -d / v
        if abs(velocity) < 1e-6:
            return np.nan
        t = -distance / velocity
        return float(t) if t > 0 else np.nan

    # Quadratic: 0.5*a*t² + v*t + d = 0
    A = 0.5 * acceleration
    B = velocity
    C = distance

    discriminant = B ** 2 - 4.0 * A * C
    if discriminant < 0:
        # No real solution — object will not reach sensor under current kinematics
        return np.nan

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B + sqrt_disc) / (2.0 * A)
    t2 = (-B - sqrt_disc) / (2.0 * A)

    candidates = [t for t in (t1, t2) if t > 0 and np.isfinite(t)]
    return float(min(candidates)) if candidates else np.nan


# ---------------------------------------------------------------------------
# Boundary erosion (unchanged)
# ---------------------------------------------------------------------------

def _erode_segment_boundaries(labels, erosion=BOUNDARY_EROSION_FRAMES):
    """
    Strip transition frames from each end of every 'towards' segment.

    Args:
        labels  (list[str]): Per-frame label strings.
        erosion (int):       Frames to remove from each boundary.

    Returns:
        list[str]: Eroded label list of the same length.
    """
    if erosion <= 0:
        return labels

    result = list(labels)
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 'towards':
            j = i
            while j < n and result[j] == 'towards':
                j += 1
            seg_len = j - i

            if seg_len <= 2 * erosion:
                for k in range(i, j):
                    result[k] = 'not_towards'
            else:
                for k in range(i, i + erosion):
                    result[k] = 'not_towards'
                for k in range(j - erosion, j):
                    result[k] = 'not_towards'
            i = j
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Main labelling pipeline
# ---------------------------------------------------------------------------

def refine_labels_by_distance():
    """
    Group raw session data and apply velocity-based binary labelling with TTC.

    Pipeline per session:
      1. Threshold velocity → raw binary labels
      2. Majority-vote smoothing (window=3)
      3. Boundary erosion
      4. Kinematic TTC via quadratic solver  ← FIX (A)
    """
    feature_dataset_path = 'data/processed/features.csv'
    output_filepath      = 'data/processed/final_labeled_data.csv'

    if not os.path.exists(feature_dataset_path):
        logger.error(f"Input file not found: {feature_dataset_path}")
        sys.exit(1)

    df             = pd.read_csv(feature_dataset_path)
    session_id_col = 'label'
    processed_groups = []

    SPEED_OF_SOUND     = 343.0
    SAMPLING_RATE      = 1953125
    VELOCITY_THRESHOLD = 0.05

    logger.info(
        f"Generating labels with velocity threshold={VELOCITY_THRESHOLD}, "
        f"boundary erosion={BOUNDARY_EROSION_FRAMES} frames, "
        f"kinematic TTC (quadratic)..."
    )

    for session_name, group_df in df.groupby(session_id_col):
        group_df = group_df.copy()
        vel = group_df['velocity']

        # Step 1 — threshold
        raw_labels = np.where(vel < -VELOCITY_THRESHOLD, 'towards', 'not_towards')

        # Step 2 — smoothing
        series   = pd.Series((raw_labels == 'towards').astype(int))
        smoothed = (
            series
            .rolling(window=3, center=True, min_periods=1)
            .apply(lambda x: int(pd.Series(x).mode().iloc[0]), raw=True)
            .ffill()
            .bfill()
            .astype(int)
        )
        motion_labels = smoothed.map({1: 'towards', 0: 'not_towards'}).tolist()

        # Step 3 — boundary erosion
        motion_labels = _erode_segment_boundaries(motion_labels, erosion=BOUNDARY_EROSION_FRAMES)
        group_df['motion'] = motion_labels

        # Step 4 — Kinematic TTC (quadratic)  FIX (A)
        group_df['calc_dist_m'] = np.where(
            group_df['echo_index'] >= 0,
            (group_df['echo_index'] / SAMPLING_RATE) * SPEED_OF_SOUND / 2.0,
            np.nan,
        )

        ttc_values = []
        for _, row in group_df.iterrows():
            if (
                row['motion'] == 'towards'
                and not np.isnan(row['calc_dist_m'])
                and abs(row['velocity']) > 0.01
            ):
                ttc = _kinematic_ttc(
                    distance=row['calc_dist_m'],
                    velocity=row['velocity'],
                    acceleration=row['acceleration'],
                )
                ttc_values.append(ttc)
            else:
                ttc_values.append(np.nan)

        group_df['ttc'] = ttc_values

        n_towards     = (group_df['motion'] == 'towards').sum()
        n_not_towards = (group_df['motion'] == 'not_towards').sum()
        logger.info(
            f"  Session '{session_name}': "
            f"towards={n_towards}, not_towards={n_not_towards}"
        )
        processed_groups.append(group_df)

    final_df = pd.concat(processed_groups)
    final_df = final_df.rename(columns={session_id_col: 'session_id', 'motion': 'label'})

    dist = final_df['label'].value_counts()
    logger.info("Final Label Distribution:")
    for label, count in dist.items():
        logger.info(f"  - {label}: {count}")

    final_df.to_csv(output_filepath, index=False)
    logger.info(f"Labeled dataset saved to {output_filepath}")


if __name__ == "__main__":
    refine_labels_by_distance()
