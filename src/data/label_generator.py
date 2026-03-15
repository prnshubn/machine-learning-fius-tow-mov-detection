"""
Label Generation & Refinement Module: Automated Ground Truth Creation.

This module converts raw sensor session data into a labeled dataset suitable 
for supervised learning. It performs three critical functions:
1. Movement Segmentation: Identifies 'Towards' movement based on velocity thresholds.
2. Signal Cleaning: Applies majority-voting and boundary erosion to remove noise.
3. Kinematic TTC: Calculates the high-precision quadratic Time-to-Collision.

Industry Approach:
- Boundary Erosion: Prevents 'label bleeding' where transition frames confuse the model.
- Quadratic Solver: Uses physics-based formulas (distance, velocity, acceleration) 
  for accurate impact prediction, outperforming simple linear models.
- Vectorized Operations: Replaces slow row-wise iteration with np.vectorize for scale.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure standardized logging for professional traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration: Remove 2 frames from each end of every movement burst
# This ensures that only clear, high-confidence 'Towards' signals are labeled.
BOUNDARY_EROSION_FRAMES = 2


def _kinematic_ttc(distance, velocity, acceleration):
    """
    Solves the kinematic equation d + v*t + 0.5*a*t^2 = 0 for impact time.

    This physics-based approach provides a more accurate ground truth than 
    simple distance/velocity, especially for accelerating objects.

    Args:
        distance (float): Current distance in meters.
        velocity (float): Current velocity in m/s (negative = approaching).
        acceleration (float): Current acceleration in m/s^2.

    Returns:
        float: Smallest positive real time to impact (TTC), or np.nan if no impact.
    """
    if distance <= 0 or np.isnan(distance):
        return np.nan

    # Linear fallback: Use simple t = -d/v if acceleration is negligible
    if abs(acceleration) < 1e-6:
        if abs(velocity) < 1e-6:
            return np.nan
        t = -distance / velocity
        return float(t) if t > 0 else np.nan

    # Quadratic coefficients for the equation: 0.5*a*t^2 + v*t + d = 0
    A = 0.5 * acceleration
    B = velocity
    C = distance

    # Calculate the discriminant (B^2 - 4AC)
    discriminant = B ** 2 - 4.0 * A * C
    if discriminant < 0:
        # No real roots: under current acceleration, the object never hits the sensor
        return np.nan

    # Solve for roots using the quadratic formula
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B + sqrt_disc) / (2.0 * A)
    t2 = (-B - sqrt_disc) / (2.0 * A)

    # We are only interested in future impacts (t > 0)
    candidates = [t for t in (t1, t2) if t > 0 and np.isfinite(t)]
    return float(min(candidates)) if candidates else np.nan


def _erode_segment_boundaries(labels, erosion=BOUNDARY_EROSION_FRAMES):
    """
    Strips transition frames from the start and end of movement bursts.

    This 'erosion' technique ensures the model trains only on the 'stable' middle 
    section of a movement, avoiding ambiguous start/stop signals.

    Args:
        labels (list[str]): Per-frame labels ('towards', 'not_towards').
        erosion (int): Number of frames to strip from each side.

    Returns:
        list[str]: The cleaned (eroded) label list.
    """
    if erosion <= 0:
        return labels

    result = list(labels)
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 'towards':
            # Identify the boundaries of the current 'towards' segment
            j = i
            while j < n and result[j] == 'towards':
                j += 1
            seg_len = j - i

            # If the segment is too short to survive erosion, discard it entirely
            if seg_len <= 2 * erosion:
                for k in range(i, j):
                    result[k] = 'not_towards'
            else:
                # Erode the transition frames at the start and end
                for k in range(i, i + erosion):
                    result[k] = 'not_towards'
                for k in range(j - erosion, j):
                    result[k] = 'not_towards'
            i = j
        else:
            i += 1

    return result


def refine_labels_by_distance():
    """
    Refines raw features into a labeled dataset with physics-based ground truth.

    Pipeline:
    1. Threshold velocity to identify directional 'Towards' motion.
    2. Apply majority-vote smoothing (window=3) to remove single-frame glitches.
    3. Apply boundary erosion to isolate the core of the movement.
    4. Solve the quadratic kinematic equation to calculate precise TTC (Vectorized).
    5. Save the finalized dataset to data/processed/final_labeled_data.csv.
    """
    feature_dataset_path = 'data/processed/features.csv'
    output_filepath      = 'data/processed/final_labeled_data.csv'

    if not os.path.exists(feature_dataset_path):
        logger.error(f"Input file not found: {feature_dataset_path}")
        sys.exit(1)

    df             = pd.read_csv(feature_dataset_path)
    session_id_col = 'label'  # Original label column from feature extractor
    processed_groups = []

    # Physics Constants
    SPEED_OF_SOUND     = 343.0    # m/s
    SAMPLING_RATE      = 1953125  # Hz
    VELOCITY_THRESHOLD = 0.05    # m/s (Directional velocity noise floor)

    logger.info(f"Generating labels: threshold={VELOCITY_THRESHOLD} m/s, erosion={BOUNDARY_EROSION_FRAMES} frames")

    # Define a wrapper for vectorization that includes the logical checks
    def _conditional_ttc(motion, dist, vel, accel):
        if motion == 'towards' and not np.isnan(dist) and abs(vel) > 0.01:
            return _kinematic_ttc(dist, vel, accel)
        return np.nan

    # Vectorize the function (specifying float output for NaNs and precision)
    vec_kinematic_ttc = np.vectorize(_conditional_ttc, otypes=[float])

    # Group by session (object type) to ensure independent processing
    for session_name, group_df in df.groupby(session_id_col):
        group_df = group_df.copy()
        vel = group_df['velocity']

        # Step 1: Initial directional thresholding
        raw_labels = np.where(vel < -VELOCITY_THRESHOLD, 'towards', 'not_towards')

        # Step 2: Majority-vote smoothing to stabilize labels
        series   = pd.Series((raw_labels == 'towards').astype(int))
        smoothed = (
            series
            .rolling(window=3, center=True, min_periods=1)
            .apply(lambda x: int(pd.Series(x).mode().iloc[0]), raw=True)
            .ffill().bfill().astype(int)
        )
        motion_labels = smoothed.map({1: 'towards', 0: 'not_towards'}).tolist()

        # Step 3: Boundary erosion (stripping transitions)
        motion_labels = _erode_segment_boundaries(motion_labels, erosion=BOUNDARY_EROSION_FRAMES)
        group_df['motion'] = motion_labels

        # Step 4: Calculate distance from echo arrival time (Time-of-Flight)
        group_df['calc_dist_m'] = np.where(
            group_df['echo_index'] >= 0,
            (group_df['echo_index'] / SAMPLING_RATE) * SPEED_OF_SOUND / 2.0,
            np.nan,
        )

        # Step 5: High-precision TTC via vectorized kinematic solver
        group_df['ttc'] = vec_kinematic_ttc(
            group_df['motion'],
            group_df['calc_dist_m'],
            group_df['velocity'],
            group_df['acceleration']
        )

        processed_groups.append(group_df)
        logger.info(f"  Processed '{session_name}': {len(group_df)} rows")

    # Consolidation and column reordering
    final_df = pd.concat(processed_groups)
    final_df = final_df.rename(columns={session_id_col: 'session_id', 'motion': 'label'})

    # Target reordering: Move 'label' to the end (ML convention)
    cols = [c for c in final_df.columns if c != 'label'] + ['label']
    final_df = final_df[cols]

    # Save to disk
    final_df.to_csv(output_filepath, index=False)
    logger.info(f"Labeled dataset saved to {output_filepath}")


if __name__ == "__main__":
    refine_labels_by_distance()