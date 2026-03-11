"""
TTC Regression Module: Predicts Time-to-Collision (TTC).
Algorithms: Linear Regression and Support Vector Regressor (SVR).
Only trains on 'towards' sequences to predict the moment of impact.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ttc_regression():
    """
    Trains regression models to predict TTC based on echo and kinematic features.
    """
    data_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Labeled data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    
    # 1. Filter for 'Towards' segments only
    towards_df = df[df['label'] == 'towards'].copy()
    if towards_df.empty:
        logger.error("No 'towards' data available for regression training.")
        sys.exit(1)

    # 2. Calculate Ground Truth TTC
    # Group by session and identify continuous approaches
    towards_df = towards_df.sort_values(by=['session_id', 'timestamp'])
    towards_df['time_diff'] = towards_df.groupby('session_id')['timestamp'].diff()
    towards_df['new_segment'] = towards_df.fillna({'time_diff': 0})['time_diff'] > 500
    towards_df['segment_id'] = towards_df.groupby('session_id')['new_segment'].cumsum()
    
    groups = towards_df.groupby(['session_id', 'segment_id'])
    
    final_rows = []
    for _, group in groups:
        if len(group) < 10: continue
        
        # Impact is assumed at the end of the 'towards' sequence
        end_time = group['timestamp'].max()
        # TTC in seconds
        group['ttc_ground_truth'] = (end_time - group['timestamp']) / 1000.0
        final_rows.append(group)
        
    regression_df = pd.concat(final_rows)
    
    # 3. Feature Selection (Using Quad-Scale naming convention)
    features = [
        'velocity', 'acceleration', 'echo_index', 
        'Peak Frequency', 'Spectral Centroid', 'Trend_50'
    ]
    X = regression_df[features]
    y = regression_df['ttc_ground_truth']
    
    # 4. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/ttc_scaler.joblib')

    # 5. Train Models
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf', C=10.0, epsilon=0.1)
    }

    results = []
    for name, model in models.items():
        logger.info(f"Training Regression Model: {name}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({"Model": name, "MAE (Seconds)": mae, "R2 Score": r2})
        joblib.dump(model, f"models/model_ttc_{name.lower().replace(' ', '_')}.joblib")

    # 6. Report and Save Best
    report_df = pd.DataFrame(results)
    logger.info("\n" + "="*40 + "\nREGRESSION PERFORMANCE (TTC)\n" + "="*40)
    print(report_df.to_string(index=False))
    
    best_name = report_df.loc[report_df['MAE (Seconds)'].idxmin()]['Model']
    joblib.dump(models[best_name], 'models/ttc_prediction_model.joblib')
    logger.info(f"Primary TTC model saved: {best_name}")

if __name__ == "__main__":
    train_ttc_regression()
