"""
Label Refinement Module: Generates binary ground truth labels (towards vs not_towards).
Captures multiple movement bursts within a single session using dynamic thresholding.
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
    """
    feature_dataset_path = 'data/processed/features.csv'
    output_filepath = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(feature_dataset_path):
        logger.error(f"Input file not found: {feature_dataset_path}")
        sys.exit(1)

    df = pd.read_csv(feature_dataset_path)
    session_id_col = 'label'
    processed_groups = []
    
    # --- INDUSTRY STRATEGY: High Sensitivity ---
    # We lower the threshold to 0.05 to capture slow movement and the start/end of bursts.
    VELOCITY_THRESHOLD = 0.05 

    logger.info(f"Generating high-sensitivity binary labels (Threshold: {VELOCITY_THRESHOLD})...")

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
        
        # We no longer "trim" the start/end of the file to ensure we don't accidentally
        # cut out a burst that happened very early or very late.
        processed_groups.append(group_df)

    final_df = pd.concat(processed_groups)
    final_df = final_df.rename(columns={session_id_col: 'session_id', 'motion': 'label'})
    
    dist = final_df['label'].value_counts()
    logger.info("Final Label Distribution (High Sensitivity):")
    for label, count in dist.items():
        logger.info(f"  - {label}: {count}")
    
    final_df.to_csv(output_filepath, index=False)
    logger.info(f"Expanded labeled dataset saved to {output_filepath}")

if __name__ == "__main__":
    refine_labels_by_distance()
