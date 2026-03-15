"""
TTC Regression Module: Predicts Time-to-Collision (TTC).
Algorithms: Linear Regression (Baseline) and Support Vector Regressor (SVR).
Uses kinematic TTC (Distance / Velocity) for training and evaluation.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ttc_regression():
    """
    Trains regression models to predict TTC based on kinematic and signal features.
    """
    data_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Labeled data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    
    # 1. Filter for valid 'Towards' segments with computed TTC
    # Drop rows where TTC is NaN or infinity
    regression_df = df[df['label'] == 'towards'].dropna(subset=['ttc']).copy()
    regression_df = regression_df[np.isfinite(regression_df['ttc'])]
    
    if regression_df.empty:
        logger.error("No valid 'towards' data with TTC available for training.")
        sys.exit(1)

    logger.info(f"Training on {len(regression_df)} valid kinematic TTC samples.")

    # 2. Feature Selection (Using Quad-Scale naming convention)
    # Using kinematics and signal features to predict TTC
    features = [
        'velocity', 'acceleration', 'echo_index', 
        'Peak Frequency', 'Spectral Centroid', 'Trend_25'
    ]
    X = regression_df[features]
    y = regression_df['ttc']
    
    # 3. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/ttc_scaler.joblib')

    # 4. Train Models
    # Linear Regression as the Baseline
    # SVR for non-linear acceleration patterns
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "SVR (Improved)": SVR(kernel='rbf', C=20.0, epsilon=0.01)
    }

    results = []
    for name, model in models.items():
        logger.info(f"Training Regression Model: {name}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Performance Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name, 
            "MAE (sec)": mae, 
            "RMSE (sec)": rmse, 
            "R2 Score": r2
        })
        
        clean_name = name.lower().split(' ')[0]
        joblib.dump(model, f"models/model_ttc_{clean_name}.joblib")

    # 5. Comparative Report
    report_df = pd.DataFrame(results)
    logger.info("\n" + "="*50 + "\nCOLLISION PREDICTION PERFORMANCE (TTC)\n" + "="*50)
    print(report_df.to_string(index=False))
    
    # Save the absolute best for production
    best_row = report_df.loc[report_df['MAE (sec)'].idxmin()]
    best_name = best_row['Model']
    
    for name, model in models.items():
        if name == best_name:
            joblib.dump(model, 'models/ttc_prediction_model.joblib')
            break
            
    logger.info(f"Final Production Model: {best_name} (MAE: {best_row['MAE (sec)']:.4f}s)")

if __name__ == "__main__":
    train_ttc_regression()
