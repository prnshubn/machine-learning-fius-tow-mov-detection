"""
Master Model Trainer: Implements the Predictive Collision Severity System.
1. Stage 1 (Classification): Detects motion towards the sensor (One-Class SVM).
2. Stage 2 (Regression): Forecasts Time-to-Collision (TTC) using Linear Regression & SVR.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_full_pipeline():
    """
    Main training function to address the exposé goals:
    - Motion Detection (towards vs not_towards)
    - TTC Prediction (Linear Regression baseline vs SVR improvement)
    """
    data_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Labeled data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    
    # --- STAGE 1: MOTION CLASSIFICATION (ONE-CLASS SVM) ---
    logger.info("--- Training Stage 1: Motion Classification ---")
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'towards' else -1)
    
    class_features = ['echo_index', 'velocity', 'acceleration', 'Peak Frequency', 'Spectral Centroid']
    X_class = df[class_features]
    y_class = df['binary_label']
    
    # Train only on 'towards' for One-Class
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    X_train_towards = X_train_c[y_train_c == 1]
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_towards)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    occ_model = OneClassSVM(kernel='rbf', nu=0.05)
    occ_model.fit(X_train_c_scaled)
    y_pred_c = occ_model.predict(X_test_c_scaled)
    
    logger.info(f"Stage 1 F1-Score: {f1_score(y_test_c, y_pred_c, pos_label=1):.4f}")
    joblib.dump(occ_model, 'models/motion_detection_model.joblib')
    joblib.dump(scaler_c, 'models/motion_scaler.joblib')


    # --- STAGE 2: TTC REGRESSION (LINEAR VS SVR) ---
    logger.info("\n--- Training Stage 2: TTC Regression ---")
    
    # Filter for valid 'towards' frames with computed kinematic TTC
    regression_df = df[df['label'] == 'towards'].dropna(subset=['ttc']).copy()
    regression_df = regression_df[np.isfinite(regression_df['ttc'])]
    
    if regression_df.empty:
        logger.error("No valid TTC data found for regression.")
        return

    # Feature selection for TTC
    reg_features = ['velocity', 'acceleration', 'echo_index', 'Peak Frequency', 'Spectral Centroid', 'Trend_25']
    X_reg = regression_df[reg_features]
    y_reg = regression_df['ttc']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    
    joblib.dump(scaler_r, 'models/ttc_scaler.joblib')

    # Models to compare
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "SVR (RBF Kernel)": SVR(kernel='rbf', C=20.0, epsilon=0.01)
    }

    results = []
    for name, model in models.items():
        logger.info(f"Fitting {name}...")
        model.fit(X_train_r_scaled, y_train_r)
        y_pred_r = model.predict(X_test_r_scaled)
        
        mae = mean_absolute_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)
        
        results.append({"Model": name, "MAE (sec)": mae, "RMSE (sec)": rmse, "R2": r2})
        
        # Save specific models
        suffix = "linear" if "Linear" in name else "svr"
        joblib.dump(model, f'models/model_ttc_{suffix}.joblib')

    # Final Comparative Report
    report_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON: PREDICTIVE COLLISION SEVERITY")
    print("="*60)
    print(report_df.to_string(index=False))
    print("="*60)
    
    # Save the absolute best for production use
    best_model_name = report_df.loc[report_df['MAE (sec)'].idxmin()]['Model']
    best_model = models[best_model_name]
    joblib.dump(best_model, 'models/ttc_prediction_model.joblib')
    logger.info(f"Primary TTC production model saved: {best_model_name}")

if __name__ == "__main__":
    train_full_pipeline()
