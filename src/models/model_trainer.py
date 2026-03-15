"""
Master Model Trainer: High-Performance Collision Prediction Pipeline.

This module implements a two-stage machine learning architecture:
Stage 1 (Classification): Detects motion towards the sensor using One-Class 
    Classification (OCC) algorithms (SVM, Isolation Forest, LOF, Autoencoders).
Stage 2 (Regression): Predicts Time-to-Collision (TTC) using Linear Regression 
    and Support Vector Regressors (SVR).

Industry Approach:
- Session-Aware Validation: Uses GroupKFold/GroupShuffleSplit to ensure the model
  generalizes to completely new recording sessions (new objects/people).
- Pipeline Architecture: Bundles preprocessing (StandardScaler) and models into 
  atomic artifacts to prevent feature-leakage and ordering bugs.
- Benchmark Comparison: Evaluates ML models against a rule-based 'Naive Baseline'.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.detectors import AutoencoderDetector

# Initialize professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System paths and constants
INPUT_PATH       = 'data/processed/final_labeled_data.csv'
MODEL_OUTPUT_DIR = 'models'
REPORTS_DIR      = 'reports'

# Quad-scale windows (must be consistent across extraction/training/inference)
WINDOWS  = [5, 10, 25, 50]
CV_N_FOLDS = 5

# Naive Baseline configuration
NAIVE_VELOCITY_THRESHOLD = -0.05
NAIVE_WINDOW             = 3


def _build_classification_features(windows):
    """
    Returns the ordered list of 17 features used for motion classification.

    Consistency check: This is the single source of truth for feature order.
    """
    features = ['echo_index', 'echo_amplitude', 'echo_width',
                'Peak Frequency', 'Spectral Centroid']
    for w in windows:
        features += [f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}']
    return features


def _make_occ_pipeline(clf):
    """
    Wraps a classifier in a scikit-learn Pipeline with a StandardScaler.

    This ensures that scaling parameters derived from training are 
    automatically and correctly applied during inference.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  clf),
    ])


def _evaluate_naive_baseline(df, test_index, y_test):
    """
    Evaluates a simple rule-based baseline for comparison.

    Rule: 'Towards' if rolling-mean velocity is below threshold.
    """
    test_velocity  = df.loc[test_index, 'velocity']
    rolling_mean_v = test_velocity.rolling(window=NAIVE_WINDOW, min_periods=1).mean()
    y_naive        = np.where(rolling_mean_v < NAIVE_VELOCITY_THRESHOLD, 1, -1)

    return {
        'Algorithm': f'Naive Baseline (vel < {NAIVE_VELOCITY_THRESHOLD})',
        'F1-Score':  f1_score(y_test, y_naive, pos_label=1, zero_division=0),
        'Accuracy':  accuracy_score(y_test, y_naive),
        'Precision': precision_score(y_test, y_naive, pos_label=1, zero_division=0),
        'Recall':    recall_score(y_test, y_naive, pos_label=1, zero_division=0),
    }


def train_master_pipeline():
    """
    Main orchestration logic for model training and validation.

    Pipeline:
    0. Validate generalization via Session-Aware Cross-Validation.
    1. Perform algorithm sweep for Stage 1 (Motion Detection).
    2. Select best classifier and generate Confusion Matrix.
    3. Train Stage 2 (TTC Prediction) on detected 'Towards' frames.
    4. Save best models as serialized Pipeline artifacts (.joblib).
    """
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Labeled data not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Label preparation: Map 'towards' to 1 and 'not_towards' to -1
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'towards' else -1)

    class_features = _build_classification_features(WINDOWS)
    logger.info(f"Classification features (17): {class_features}")

    X_class  = df[class_features].fillna(0)
    y_class  = df['binary_label']
    sessions = df['session_id'].values

    # -----------------------------------------------------------------------
    # STAGE 0 — SESSION-AWARE CROSS-VALIDATION
    # -----------------------------------------------------------------------
    logger.info("--- Stage 0: Session-Aware Cross-Validation ---")

    n_unique_sessions = df['session_id'].nunique()
    n_folds           = min(CV_N_FOLDS, n_unique_sessions)
    cv_run            = False
    cv_f1, cv_prec, cv_rec = [], [], []

    if n_folds < 2:
        logger.warning(f"Only {n_unique_sessions} session(s) found. Skipping CV.")
    else:
        cv_run = True
        gkf    = GroupKFold(n_splits=n_folds)

        # Cross-validate using One-Class SVM as the reference model
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_class, y_class, groups=sessions)):
            X_tr = X_class.iloc[train_idx]
            y_tr = y_class.iloc[train_idx]
            X_te = X_class.iloc[test_idx]
            y_te = y_class.iloc[test_idx]

            # OCC models train only on the positive class ('towards')
            X_tr_towards = X_tr[y_tr == 1]
            if len(X_tr_towards) < 5:
                continue

            cv_pipe = _make_occ_pipeline(OneClassSVM(kernel='rbf', nu=0.05))
            cv_pipe.fit(X_tr_towards)
            y_cv_pred = cv_pipe.predict(X_te)

            cv_f1.append(  f1_score(   y_te, y_cv_pred, pos_label=1, zero_division=0))
            cv_prec.append(precision_score(y_te, y_cv_pred, pos_label=1, zero_division=0))
            cv_rec.append( recall_score(   y_te, y_cv_pred, pos_label=1, zero_division=0))
            logger.info(f"  Fold {fold + 1} | Sessions held out: {np.unique(sessions[test_idx]).tolist()} | F1={cv_f1[-1]:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 1 — OCC ALGORITHM SWEEP
    # -----------------------------------------------------------------------
    logger.info("--- Stage 1: OCC Algorithm Sweep ---")

    # Use a stratified split for the final evaluation
    from sklearn.model_selection import train_test_split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, stratify=y_class, random_state=42
    )
    X_train_towards = X_train_c[y_train_c == 1]

    # Evaluate multiple One-Class algorithms
    occ_pipelines = {
        "One-Class SVM (nu=0.01)":    _make_occ_pipeline(OneClassSVM(kernel='rbf', nu=0.01)),
        "One-Class SVM (nu=0.05)":    _make_occ_pipeline(OneClassSVM(kernel='rbf', nu=0.05)),
        "Isolation Forest (100 est)": _make_occ_pipeline(IsolationForest(n_estimators=100,  contamination=0.1, random_state=42)),
        "Isolation Forest (200 est)": _make_occ_pipeline(IsolationForest(n_estimators=200,  contamination=0.1, random_state=42)),
        "Local Outlier Factor":       _make_occ_pipeline(LocalOutlierFactor(n_neighbors=20, novelty=True)),
        "Autoencoder (Narrow)":       _make_occ_pipeline(AutoencoderDetector(hidden_layer_sizes=(4, 2, 4))),
        "Autoencoder (Wide)":         _make_occ_pipeline(AutoencoderDetector(hidden_layer_sizes=(8, 4, 8))),
    }

    detailed_results = []
    for name, pipe in occ_pipelines.items():
        logger.info(f"  Training: {name}")
        pipe.fit(X_train_towards)
        y_pred = pipe.predict(X_test_c)

        detailed_results.append({
            "Algorithm": name,
            "F1-Score":  f1_score(  y_test_c, y_pred, pos_label=1, zero_division=0),
            "Accuracy":  accuracy_score(y_test_c, y_pred),
            "Precision": precision_score(y_test_c, y_pred, pos_label=1, zero_division=0),
            "Recall":    recall_score(  y_test_c, y_pred, pos_label=1, zero_division=0),
        })

        # Save individual model snapshots
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        joblib.dump(pipe, f'models/pipeline_{clean_name}.joblib')

    detailed_df = pd.DataFrame(detailed_results)

    # Evaluate Baseline
    naive_result = _evaluate_naive_baseline(df, X_test_c.index, y_test_c)
    full_comparison_df = pd.concat([pd.DataFrame([naive_result]), detailed_df], ignore_index=True)
    full_comparison_df.to_csv(os.path.join(REPORTS_DIR, 'detailed_algorithm_performance.csv'), index=False)

    # Identify and save the best classification model
    best_idx       = detailed_df['F1-Score'].idxmax()
    best_algo_name = detailed_df.iloc[best_idx]['Algorithm']
    best_f1        = detailed_df.iloc[best_idx]['F1-Score']
    logger.info(f"Best algorithm: {best_algo_name} (F1={best_f1:.4f})")

    best_pipe    = occ_pipelines[best_algo_name]
    y_pred_best  = best_pipe.predict(X_test_c)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_best, labels=[1, -1])
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(['Actual: towards', 'Actual: not_towards'], name='Actual'),
        columns=pd.Index(['Pred: towards', 'Pred: not_towards'], name='Predicted'),
    )
    cm_df.to_csv(os.path.join(REPORTS_DIR, 'confusion_matrix.csv'))

    joblib.dump(best_pipe, 'models/motion_detection_model.joblib')
    logger.info(f"Production motion pipeline saved: {best_algo_name}")

    # -----------------------------------------------------------------------
    # STAGE 2 — TTC REGRESSION
    # -----------------------------------------------------------------------
    logger.info("\n--- Stage 2: TTC Regression ---")

    # TTC is only relevant for 'towards' movement frames
    regression_df = df[df['label'] == 'towards'].dropna(subset=['ttc']).copy()
    regression_df = regression_df[np.isfinite(regression_df['ttc'])]

    if regression_df.empty:
        logger.error("No valid TTC data for regression.")
        return

    reg_features = [
        'velocity', 'acceleration', 'echo_index',
        'Peak Frequency', 'Spectral Centroid', 'Trend_25',
    ]
    X_reg     = regression_df[reg_features]
    y_reg     = regression_df['ttc']
    reg_sessions = regression_df['session_id'].values

    # Session-aware split for regression: ensures objects in test are never in train
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_r, test_idx_r = next(gss.split(X_reg, y_reg, groups=reg_sessions))

    X_train_r, X_test_r = X_reg.iloc[train_idx_r], X_reg.iloc[test_idx_r]
    y_train_r, y_test_r = y_reg.iloc[train_idx_r], y_reg.iloc[test_idx_r]

    # TTC Regression Pipelines
    reg_pipelines = {
        "Linear Regression (Baseline)": _make_occ_pipeline(LinearRegression()),
        "SVR (RBF Kernel)":             _make_occ_pipeline(SVR(kernel='rbf', C=20.0, epsilon=0.01)),
    }

    reg_results = []
    for name, pipe in reg_pipelines.items():
        logger.info(f"  Fitting {name}...")
        pipe.fit(X_train_r, y_train_r)
        y_pred_r = pipe.predict(X_test_r)

        reg_results.append({
            "Model": name,
            "MAE (sec)": mean_absolute_error(y_test_r, y_pred_r),
            "RMSE (sec)": np.sqrt(mean_squared_error(y_test_r, y_pred_r)),
            "R2": r2_score(y_test_r, y_pred_r)
        })

    reg_df = pd.DataFrame(reg_results)
    
    # Save the best regression model (lowest MAE)
    best_ttc_idx  = reg_df['MAE (sec)'].idxmin()
    best_ttc_name = reg_results[best_ttc_idx]['Model']
    joblib.dump(reg_pipelines[best_ttc_name], 'models/ttc_prediction_model.joblib')
    logger.info(f"Production TTC pipeline: {best_ttc_name}")


if __name__ == "__main__":
    train_master_pipeline()
