"""
Label Refinement Module: Generates binary ground truth labels (towards vs not_towards).
Captures multiple movement bursts within a single session using dynamic thresholding.
Now includes kinematic TTC (Time-to-Collision) calculation.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def refine_labels_by_distance():
    """
    Groups raw session data and applies sensitive velocity-based binary labeling.
    Captures all "towards" segments by using a lower noise floor.
    Calculates Kinematic TTC: Distance / |Velocity|.
    """
    feature_dataset_path = 'data/processed/features.csv'
    output_filepath = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(feature_dataset_path):
        logger.error(f"Input file not found: {feature_dataset_path}")
        sys.exit(1)

    df = pd.read_csv(feature_dataset_path)
    session_id_col = 'label'
    processed_groups = []
    
    # Constants for TTC
    SPEED_OF_SOUND = 343.0 # m/s
    SAMPLING_RATE = 1953125 # Hz
    VELOCITY_THRESHOLD = 0.05 

    logger.info(f"Generating high-sensitivity binary labels and TTC (Threshold: {VELOCITY_THRESHOLD})...")

    for session_name, group_df in df.groupby(session_id_col):
        # Physical Velocity from Kalman Filter
        vel = group_df['velocity']

        # Any negative velocity (getting closer) above the noise floor
        conditions = [vel < -VELOCITY_THRESHOLD, vel >= -VELOCITY_THRESHOLD]
        choices = [0, 1] 
        raw_labels = np.select(conditions, choices, default=1)

        # Small window smoothing to preserve short bursts
        series = pd.Series(raw_labels)
        smoothed = series.rolling(window=3, center=True, min_periods=1).apply(
            lambda x: pd.Series(x).mode().iloc[0] if not x.empty else 1
        ).ffill().bfill().values

        group_df['motion'] = pd.Series(smoothed).astype(int).map({0: 'towards', 1: 'not_towards'}).values
        
        # --- TTC CALCULATION (Kinematic) ---
        # Guard: Only calculate distance if a valid echo was detected (index >= 0)
        group_df['calc_dist_m'] = np.where(
            group_df['echo_index'] >= 0,
            (group_df['echo_index'] / SAMPLING_RATE) * SPEED_OF_SOUND / 2.0,
            np.nan
        )
        
        # TTC = distance / |velocity|
        # Only valid if moving towards (vel < 0) and distance is valid
        group_df['ttc'] = np.where(
            (group_df['motion'] == 'towards') & (np.abs(group_df['velocity']) > 0.01) & (~group_df['calc_dist_m'].isna()),
            group_df['calc_dist_m'] / np.abs(group_df['velocity']),
            np.nan
        )

        processed_groups.append(group_df)

    final_df = pd.concat(processed_groups)
    final_df = final_df.rename(columns={session_id_col: 'session_id', 'motion': 'label'})
    
    dist = final_df['label'].value_counts()
    logger.info("Final Label Distribution:")
    for label, count in dist.items():
        logger.info(f"  - {label}: {count}")
    
    final_df.to_csv(output_filepath, index=False)
    logger.info(f"Labeled dataset with TTC saved to {output_filepath}")

if __name__ == "__main__":
    refine_labels_by_distance()
