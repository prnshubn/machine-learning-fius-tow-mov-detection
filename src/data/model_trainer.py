"""
Model Training Module: Trains and evaluates regression models to predict Time-to-Collision (TTC).
Implements a Linear Regression baseline and an SVR model as specified in the exposé.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
INPUT_PATH = 'data/processed/final_labeled_data.csv'
MODEL_OUTPUT_DIR = 'models'

# Feature columns produced by the feature engineering pipeline
FEATURE_COLUMNS = [
    'velocity',
    'acceleration',
    'echo_index',
    'Peak Frequency',
    'Spectral Centroid',
    'Trend_5',
    'Mean_5',
    'Centroid_Trend_5',
    'Trend_10',
    'Mean_10',
    'Centroid_Trend_10',
    'Trend_25',
    'Mean_25',
    'Centroid_Trend_25',
    'Trend_50',
    'Mean_50',
    'Centroid_Trend_50',
]
TARGET_COLUMN = 'ttc'


def load_training_data(filepath):
    """
    Loads final_labeled_data.csv and filters to valid TTC rows only.
    TTC is only defined for 'towards' frames with a valid echo detection.

    Args:
        filepath (str): Path to the labeled dataset CSV.

    Returns:
        tuple: (X, y) as numpy arrays ready for training.
    """
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} total rows from {filepath}")

    # Keep only rows where TTC is a valid, finite number
    df_ttc = df[df[TARGET_COLUMN].notna() & np.isfinite(df[TARGET_COLUMN])].copy()
    logger.info(f"Rows with valid TTC (towards + valid echo): {len(df_ttc)}")

    if len(df_ttc) < 20:
        logger.error("Not enough valid TTC samples to train. Check your pipeline output.")
        sys.exit(1)

    # Drop any feature columns that are missing
    available_features = [c for c in FEATURE_COLUMNS if c in df_ttc.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        logger.warning(f"Missing feature columns (will be skipped): {missing}")

    X = df_ttc[available_features].fillna(0).values
    y = df_ttc[TARGET_COLUMN].values

    logger.info(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")
    logger.info(f"TTC range: {y.min():.4f}s — {y.max():.4f}s | Mean: {y.mean():.4f}s")

    return X, y, available_features


def evaluate_model(name, model, X_test, y_test):
    """
    Prints RMSE and MAE for a fitted model on the test set.

    Args:
        name (str): Display name of the model.
        model: Fitted sklearn pipeline or model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True TTC values.

    Returns:
        dict: {'rmse': float, 'mae': float}
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"  {name:<30} RMSE: {rmse:.4f}s   MAE: {mae:.4f}s")
    return {'rmse': rmse, 'mae': mae}


def train_models():
    """
    Full training pipeline:
      1. Load and filter data
      2. Train/test split (80/20, stratification not needed for regression)
      3. Fit Linear Regression baseline
      4. Fit SVR with RBF kernel
      5. Evaluate and compare both models
      6. Save both models to disk
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    X, y, feature_names = load_training_data(INPUT_PATH)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Linear Regression baseline (wrapped in a StandardScaler pipeline)
    #    Scaling is good practice even for linear models when feature magnitudes differ.
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    logger.info("Training Linear Regression baseline...")
    lr_pipeline.fit(X_train, y_train)

    # 4. SVR with RBF kernel
    #    SVR is sensitive to feature scale — StandardScaler is essential.
    #    C=10 and epsilon=0.01 are reasonable starting points for TTC in seconds.
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=10.0, epsilon=0.01, gamma='scale'))
    ])
    logger.info("Training SVR (RBF kernel)...")
    svr_pipeline.fit(X_train, y_train)

    # 5. Evaluation
    logger.info("=" * 55)
    logger.info("MODEL EVALUATION ON TEST SET")
    logger.info("=" * 55)
    lr_scores = evaluate_model("Linear Regression", lr_pipeline, X_test, y_test)
    svr_scores = evaluate_model("SVR (RBF kernel)", svr_pipeline, X_test, y_test)
    logger.info("=" * 55)

    winner = "SVR" if svr_scores['rmse'] < lr_scores['rmse'] else "Linear Regression"
    improvement = abs(lr_scores['rmse'] - svr_scores['rmse'])
    logger.info(f"Best model: {winner} (RMSE improvement over baseline: {improvement:.4f}s)")

    # 6. Save models
    lr_path = os.path.join(MODEL_OUTPUT_DIR, 'linear_regression_ttc.joblib')
    svr_path = os.path.join(MODEL_OUTPUT_DIR, 'svr_ttc.joblib')
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(svr_pipeline, svr_path)
    logger.info(f"Saved Linear Regression model to: {lr_path}")
    logger.info(f"Saved SVR model to:               {svr_path}")

    return lr_pipeline, svr_pipeline


if __name__ == "__main__":
    train_models()
