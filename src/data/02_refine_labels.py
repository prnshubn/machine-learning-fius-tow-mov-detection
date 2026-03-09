"""
This script refines the dataset by generating accurate, row-by-row motion labels.
It loads the feature set, calculates the change in distance for each measurement,
and assigns a 'approaching', 'receding', or 'stationary' label to each row.
"""

# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import os           # For file path operations
import sys

def refine_labels_by_distance():
    """
    Generates precise motion labels based on the change in the 'distance' feature.
    """
    # --- File Paths ---
    feature_dataset_path = 'data/processed/features.csv'  # Input feature dataset
    output_filepath = 'data/processed/final_labeled_data.csv'  # Output file for labeled data
    
    # Check if the feature dataset exists
    if not os.path.exists(feature_dataset_path):
        print(f"Error: The feature dataset '{feature_dataset_path}' was not found.")
        print("Please run 'python3 src/data/01_build_features.py' first to create it.")
        sys.exit(1)

    # --- Fast Efficiency Check: Modification Time ---
    if os.path.exists(output_filepath):
        # If output is newer than input, skip
        input_mtime = os.path.getmtime(feature_dataset_path)
        output_mtime = os.path.getmtime(output_filepath)
        
        if output_mtime > input_mtime:
            print(f"Step 2: '{output_filepath}' is up to date (Fast mtime check). Skipping.")
            return
        else:
            print(f"Step 2: Input features have changed. Refining labels...")

    # Load the feature dataset into a DataFrame
    print(f"Loading feature dataset from '{feature_dataset_path}'...")
    df = pd.read_csv(feature_dataset_path)
    
    # The original 'label' column identifies the source file (e.g., 'metal_plate').
    # We'll use it to group the data, as each file is a separate, continuous session.
    session_id_col = 'label'
    
    print("Generating accurate motion labels based on distance changes...")
    
    # List to hold the processed dataframes for each session
    processed_groups = []
    
    # Define a noise threshold (in meters/units) for stationary detection
    DISTANCE_THRESHOLD = 0.002 
    
    # Group by the session ID to process each experiment independently
    for session_name, group_df in df.groupby(session_id_col):
        # Calculate the difference in distance from the previous measurement
        distance_diff = group_df['distance'].diff()
        
        # Define the conditions for our new motion labels
        conditions = [
            distance_diff < -DISTANCE_THRESHOLD,  # Approaching (negative change)
            distance_diff > DISTANCE_THRESHOLD,   # Receding (positive change)
            (distance_diff >= -DISTANCE_THRESHOLD) & (distance_diff <= DISTANCE_THRESHOLD) # Stationary
        ]
        
        # Define the corresponding labels for each condition
        choices = ['approaching', 'receding', 'stationary']
        
        # Create the new 'motion' column using np.select
        group_df['motion'] = np.select(conditions, choices, default='stationary')
        
        # Add the processed group to the list
        processed_groups.append(group_df)
        
    # Combine all the processed groups back into a single dataframe
    final_df = pd.concat(processed_groups)
    
    # --- Prepare Final Dataset for Training ---
    # Rename for clarity
    final_df = final_df.rename(columns={session_id_col: 'session_id'})
    final_df = final_df.rename(columns={'motion': 'label'})
    
    # Print the class distribution
    print("--- New Dataset Class Distribution ---")
    print(final_df['label'].value_counts())
    print("--------------------------------------")
    
    # --- Save the Final Dataset ---
    try:
        final_df.to_csv(output_filepath, index=False)
        print(f"Successfully created the final labeled dataset!")
        print(f"It has been saved to: '{output_filepath}'")
    except Exception as e:
        print(f"An error occurred while saving the final dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the label refinement process when the script is executed directly
    refine_labels_by_distance()
