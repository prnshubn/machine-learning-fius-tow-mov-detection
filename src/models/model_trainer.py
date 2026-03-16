"""
Master Model Trainer: High-Performance Collision Prediction Pipeline.

This module implements a three-stage machine learning architecture:
Stage 1 (Classification): Detects motion towards the sensor using One-Class 
    Classification (OCC) algorithms: OCSVM, Isolation Forest, LOF, Autoencoder.
Stage 2 (Regression): Predicts Time-to-Collision (TTC) using SVR.
Stage 3 (Classification): Predicts Material Type to estimate Impact Force.

Industry Approach:
- Session-Aware Validation: Uses GroupKFold/GroupShuffleSplit.
- Pipeline Architecture: Bundles preprocessing (StandardScaler) and models into 
  atomic artifacts to prevent feature-leakage and scaling issues (critical for SVR).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.detectors import AutoencoderDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH       = 'data/processed/final_labeled_data.csv'
MODEL_OUTPUT_DIR = 'models'
REPORTS_DIR      = 'reports'

WINDOWS  = [5, 10, 25, 50]
CV_N_FOLDS = 5
NAIVE_VELOCITY_THRESHOLD = -0.05
NAIVE_WINDOW             = 3


def _build_classification_features(windows):
    features = ['echo_index', 'echo_amplitude', 'echo_width',
                'Peak Frequency', 'Spectral Centroid']
    for w in windows:
        features += [f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}']
    return features


def _make_pipeline(clf):
    """
    Wraps an estimator in a scikit-learn Pipeline with a StandardScaler.
    Critically important for distance-based algorithms like SVR and SVM.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  clf),
    ])


def _evaluate_naive_baseline(df, test_index, y_test):
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
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Labeled data not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'towards' else -1)
    class_features = _build_classification_features(WINDOWS)

    X_class  = df[class_features].fillna(0)
    y_class  = df['binary_label']
    sessions = df['session_id'].values

    # -----------------------------------------------------------------------
    # STAGE 0 — SESSION-AWARE CROSS-VALIDATION
    # -----------------------------------------------------------------------
    logger.info("--- Stage 0: Session-Aware Cross-Validation ---")
    n_unique_sessions = df['session_id'].nunique()
    n_folds           = min(CV_N_FOLDS, n_unique_sessions)

    if n_folds >= 2:
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_class, y_class, groups=sessions)):
            X_tr, y_tr = X_class.iloc[train_idx], y_class.iloc[train_idx]
            X_te, y_te = X_class.iloc[test_idx], y_class.iloc[test_idx]

            X_tr_towards = X_tr[y_tr == 1]
            if len(X_tr_towards) < 5: continue

            cv_pipe = _make_pipeline(OneClassSVM(kernel='rbf', nu=0.05))
            cv_pipe.fit(X_tr_towards)
            y_cv_pred = cv_pipe.predict(X_te)
            f1 = f1_score(y_te, y_cv_pred, pos_label=1, zero_division=0)
            logger.info(f"  Fold {fold + 1} | Sessions held out: {np.unique(sessions[test_idx]).tolist()} | F1={f1:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 1 — OCC ALGORITHM SWEEP
    # -----------------------------------------------------------------------
    logger.info("--- Stage 1: OCC Algorithm Sweep ---")
    gss_class = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_c, test_idx_c = next(gss_class.split(X_class, y_class, groups=sessions))
    
    X_train_c, X_test_c = X_class.iloc[train_idx_c], X_class.iloc[test_idx_c]
    y_train_c, y_test_c = y_class.iloc[train_idx_c], y_class.iloc[test_idx_c]
    X_train_towards = X_train_c[y_train_c == 1]

    # LocalOutlierFactor requires novelty=True to expose a predict() method on
    # unseen data. Without it, LOF only supports fit_predict() on the training
    # set itself and raises NotFittedError when called on X_test_c.
    occ_pipelines = {
        "One-Class SVM (nu=0.05)":    _make_pipeline(OneClassSVM(kernel='rbf', nu=0.05)),
        "Isolation Forest (100 est)": _make_pipeline(IsolationForest(n_estimators=100, contamination=0.1, random_state=42)),
        "LOF (k=20, novelty)":        _make_pipeline(LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)),
        "Autoencoder (Narrow)":       _make_pipeline(AutoencoderDetector(hidden_layer_sizes=(4, 2, 4))),
    }

    ml_results = []
    for name, pipe in occ_pipelines.items():
        pipe.fit(X_train_towards)
        y_pred = pipe.predict(X_test_c)
        ml_results.append({
            "Algorithm": name,
            "F1-Score":  f1_score(y_test_c, y_pred, pos_label=1, zero_division=0),
            "Accuracy":  accuracy_score(y_test_c, y_pred),
            "Precision": precision_score(y_test_c, y_pred, pos_label=1, zero_division=0),
            "Recall":    recall_score(y_test_c, y_pred, pos_label=1, zero_division=0),
        })

    # --- Best model selection (ML pipelines only, before baseline is appended) ---
    # The naive baseline must NOT participate in idxmax() — it is not a key in
    # occ_pipelines, so picking it would cause a KeyError on the next line.
    ml_df = pd.DataFrame(ml_results)
    best_idx = ml_df['F1-Score'].idxmax()
    best_algo_name = ml_df.iloc[best_idx]['Algorithm']
    logger.info(f"Best motion algorithm: {best_algo_name} (F1={ml_df.iloc[best_idx]['F1-Score']:.4f})")

    # Save best model and its confusion matrix.
    best_pipe = occ_pipelines[best_algo_name]
    joblib.dump(best_pipe, 'models/motion_detection_model.joblib')

    y_pred_best = best_pipe.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred_best, labels=[1, -1])
    cm_df = pd.DataFrame(cm, index=['Actual: towards', 'Actual: not towards'],
                         columns=['Predicted: towards', 'Predicted: not towards'])
    cm_df.to_csv(os.path.join(REPORTS_DIR, 'confusion_matrix.csv'), index_label='Motion Label')

    # Append naive baseline AFTER selection so it appears in the report but
    # cannot be chosen as the production model.
    test_index_labels = X_class.iloc[test_idx_c].index
    baseline_row = _evaluate_naive_baseline(df, test_index_labels, y_test_c)
    logger.info(f"Naive baseline F1={baseline_row['F1-Score']:.4f} (benchmark for ML uplift)")
    ml_results.append(baseline_row)

    pd.DataFrame(ml_results).to_csv(
        os.path.join(REPORTS_DIR, 'detailed_algorithm_performance.csv'), index=False
    )
    logger.info("Saved performance reports and confusion matrix.")

    # -----------------------------------------------------------------------
    # STAGE 2 — TTC REGRESSION
    # -----------------------------------------------------------------------
    logger.info("\n--- Stage 2: TTC Regression ---")
    regression_df = df[df['label'] == 'towards'].dropna(subset=['ttc']).copy()
    regression_df = regression_df[np.isfinite(regression_df['ttc'])]

    if not regression_df.empty:
        reg_features = ['velocity', 'acceleration', 'echo_index', 'Peak Frequency', 'Spectral Centroid', 'Trend_25']
        X_reg, y_reg = regression_df[reg_features], regression_df['ttc']
        reg_sessions = regression_df['session_id'].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx_r, test_idx_r = next(gss.split(X_reg, y_reg, groups=reg_sessions))
        X_train_r, X_test_r = X_reg.iloc[train_idx_r], X_reg.iloc[test_idx_r]
        y_train_r, y_test_r = y_reg.iloc[train_idx_r], y_reg.iloc[test_idx_r]

        reg_pipelines = {
            "Linear Regression": _make_pipeline(LinearRegression()),
            "SVR (RBF Kernel)":  _make_pipeline(SVR(kernel='rbf', C=20.0, epsilon=0.01)),
        }

        reg_results = []
        for name, pipe in reg_pipelines.items():
            pipe.fit(X_train_r, y_train_r)
            y_pred_r = pipe.predict(X_test_r)
            reg_results.append({
                "Model": name,
                "MAE (sec)": mean_absolute_error(y_test_r, y_pred_r),
                "RMSE (sec)": np.sqrt(mean_squared_error(y_test_r, y_pred_r)),
            })
        
        reg_df = pd.DataFrame(reg_results)
        best_ttc_name = reg_df.loc[reg_df['MAE (sec)'].idxmin(), 'Model']
        joblib.dump(reg_pipelines[best_ttc_name], 'models/ttc_prediction_model.joblib')
        logger.info(f"Production TTC pipeline: {best_ttc_name}")

    # -----------------------------------------------------------------------
    # STAGE 3 — MATERIAL CLASSIFICATION (For Impact Force)
    # -----------------------------------------------------------------------
    logger.info("\n--- Stage 3: Material Classification ---")
    
    mat_features = ['echo_amplitude', 'echo_width', 'Spectral Centroid', 'Peak Frequency']
    X_mat = df[mat_features]
    y_mat = df['session_id'] # e.g., 'metal_plate', 'human', 'cardboard'
    
    # We use Stratified split to ensure we learn the signature of all known materials
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mat, y_mat, test_size=0.2, random_state=42, stratify=y_mat
    )
    
    mat_pipeline = _make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    mat_pipeline.fit(X_train_m, y_train_m)
    y_pred_m = mat_pipeline.predict(X_test_m)
    
    acc = accuracy_score(y_test_m, y_pred_m)
    logger.info(f"Material Classification Accuracy: {acc:.4f}")
    
    joblib.dump(mat_pipeline, 'models/material_classifier_model.joblib')
    logger.info("Production Material pipeline saved.")

if __name__ == "__main__":
    train_master_pipeline()