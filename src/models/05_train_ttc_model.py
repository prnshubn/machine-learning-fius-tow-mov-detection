
"""
This script trains a regression model to predict Time-To-Collision (TTC).
It uses the 'approaching' segments of the data, calculates the ground truth TTC,
and trains Linear Regression and SVR models.

Main functionalities:
- Load labeled data
- Filter for 'approaching' segments
- Calculate ground truth TTC for each segment
- Train regression models (Linear Regression, SVR)
- Evaluate and save the best model and scaler
"""

# Import required libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression  # Regression model
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Evaluation metrics
from sklearn.preprocessing import StandardScaler  # For feature scaling
import joblib  # For saving models
import os  # For file operations
import warnings  # To suppress warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def train_ttc_model():
    """
    Trains and evaluates TTC regression models.
    Steps:
    1. Load labeled data
    2. Filter for 'approaching' segments
    3. Identify continuous approach events (segments)
    4. Calculate ground truth TTC for each row
    5. Prepare features and targets
    6. Train and evaluate regression models
    7. Save the best model and scaler
    """
    data_path = 'data/processed/final_labeled_data.csv'  # Path to labeled data
    
    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        return

    # Load the labeled data
    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # --- Filter for Approaching Data ---
    # We only care about TTC when the object is approaching
    approaching_df = df[df['label'] == 'approaching'].copy()
    
    # Check if there is any approaching data
    if approaching_df.empty:
        print("Error: No 'approaching' data found to train TTC model.")
        return

    # --- Calculate Ground Truth TTC ---
    # Identify continuous approaching sequences to define the "impact" time (T=0)
    # Group by session_id and contiguous label blocks
    # Sort by session and timestamp for proper sequence
    approaching_df = approaching_df.sort_values(by=['session_id', 'timestamp'])
    
    # Since we already filtered for 'approaching', we need to distinguish between different approach events
    # in the same session (e.g. approach -> recede -> approach).
    # A simple heuristic: if timestamp diff > threshold (e.g. 500ms), it's a new segment.
    approaching_df['time_diff'] = approaching_df.groupby('session_id')['timestamp'].diff()
    approaching_df['new_segment'] = approaching_df['time_diff'] > 500
    approaching_df['segment_id'] = approaching_df.groupby('session_id')['new_segment'].cumsum()
    
    # Now group by session and segment
    groups = approaching_df.groupby(['session_id', 'segment_id'])
    
    ttc_list = []
    
    for _, group in groups:
        if len(group) < 5: # Skip very short segments
            ttc_list.extend([np.nan] * len(group))
            continue
            
        # The end of the approaching sequence is assumed to be impact or closest point
        # Timestamp is likely in milliseconds (based on ~45ms delta)
        end_time = group['timestamp'].max()
        
        # TTC = (End Time - Current Time) / 1000.0 (to get seconds)
        ttc = (end_time - group['timestamp']) / 1000.0
        ttc_list.extend(ttc)
        
    approaching_df['ttc_ground_truth'] = ttc_list
    
    # Drop rows with NaN TTC (short segments)
    approaching_df = approaching_df.dropna(subset=['ttc_ground_truth'])
    
    print(f"Training on {len(approaching_df)} samples from approaching sequences.")
    
    # --- Feature Selection ---
    # We use distance, velocity, acceleration, and spectral features
    features = [
        'distance', 'velocity', 'acceleration', 
        'Peak Frequency', 'Mean Frequency', 'Spectral Centroid', 
        'Spectral Skewness', 'Spectral Kurtosis', 'first_peak_index'
    ]
    
    # Check if all features exist
    available_features = [f for f in features if f in approaching_df.columns]
    
    X = approaching_df[available_features]
    y = approaching_df['ttc_ground_truth']
    
    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Model 1: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print("\n--- Linear Regression Results ---")
    print(f"MAE: {mae_lr:.4f} seconds")
    print(f"R2 Score: {r2_lr:.4f}")
    
    # --- Model 2: SVR ---
    print("\n--- Training SVR (this may take a moment) ---")
    svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    
    print("--- SVR Results ---")
    print(f"MAE: {mae_svr:.4f} seconds")
    print(f"R2 Score: {r2_svr:.4f}")
    
    # --- Select Best Model ---
    if mae_svr < mae_lr:
        best_model = svr
        name = "SVR"
        print("\nSVR performed better.")
    else:
        best_model = lr
        name = "Linear Regression"
        print("\nLinear Regression performed better.")
        
    # --- Save Model ---
    model_save_path = 'models/ttc_model.joblib'
    scaler_save_path = 'models/ttc_scaler.joblib'
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"Saved best TTC model ({name}) to '{model_save_path}'")
    print(f"Saved scaler to '{scaler_save_path}'")

if __name__ == "__main__":
    train_ttc_model()
