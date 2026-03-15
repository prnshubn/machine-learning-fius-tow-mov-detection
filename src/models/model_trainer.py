"""
Master Model Trainer: Implements the Predictive Collision Severity System.

Stage 1 — Classification : Detects motion towards the sensor.
           Algorithm sweep: One-Class SVM, Isolation Forest, LOF, Autoencoder.
Stage 2 — Regression     : Forecasts Time-to-Collision (TTC).
           Models: Linear Regression (baseline) and SVR (improvement).

FIX (D) — sklearn Pipeline:
    Every model is now wrapped in a Pipeline([('scaler', StandardScaler()),
    ('model', ...)]).  The Pipeline is saved as a single .joblib artifact.
    predictor.py loads the Pipeline and calls .predict() directly — it never
    needs to manage a separate scaler, so feature-ordering bugs become
    impossible: the scaler was fitted on exactly the columns the model expects.

FIX (B3) — TTC regression data leakage:
    The plain train_test_split is replaced by GroupShuffleSplit keyed on
    session_id. Frames from the same recording session can no longer appear
    in both train and test, so the reported MAE/R² reflect genuine
    generalisation to unseen sessions.
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
    """
    Single source of truth for the 17-column classification feature list.

    Layout:
      [echo_index, echo_amplitude, echo_width, Peak Frequency, Spectral Centroid]  — 5
      + [Trend_W, Mean_W, Centroid_Trend_W] × len(windows)                        — 12
      = 17 total
    """
    features = ['echo_index', 'echo_amplitude', 'echo_width',
                'Peak Frequency', 'Spectral Centroid']
    for w in windows:
        features += [f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}']
    return features


def _make_occ_pipeline(clf):
    """
    Wrap a one-class classifier in a Pipeline with StandardScaler.

    FIX (D): The Pipeline is the single saved artifact. predictor.py loads it
    and calls pipeline.predict(X_raw) — no external scaler needed.

    Note: OCC pipelines are fitted on X_train_towards only (positive class).
    Calling pipeline.predict(X_test) scales with that same positive-class
    statistics, which is correct because the scaler was fitted on the same
    distribution the model learned from.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  clf),
    ])


def _evaluate_naive_baseline(df, test_index, y_test):
    """Rule-based baseline: 'towards' if rolling-mean velocity < threshold."""
    test_velocity  = df.loc[test_index, 'velocity']
    rolling_mean_v = test_velocity.rolling(window=NAIVE_WINDOW, min_periods=1).mean()
    y_naive        = np.where(rolling_mean_v < NAIVE_VELOCITY_THRESHOLD, 1, -1)

    return {
        'Algorithm': f'Naive Baseline (vel < {NAIVE_VELOCITY_THRESHOLD} over {NAIVE_WINDOW} frames)',
        'F1-Score':  f1_score(y_test, y_naive, pos_label=1, zero_division=0),
        'Accuracy':  accuracy_score(y_test, y_naive),
        'Precision': precision_score(y_test, y_naive, pos_label=1, zero_division=0),
        'Recall':    recall_score(y_test, y_naive, pos_label=1, zero_division=0),
    }


def train_master_pipeline():
    """
    Full training pipeline:
      0. Session-aware cross-validation (GroupKFold)
      1. OCC algorithm sweep vs naive baseline on a stratified hold-out set
      2. Confusion matrix for the best ML algorithm
      3. TTC regression with session-aware split (GroupShuffleSplit)  FIX (B3)

    All models saved as sklearn Pipelines (scaler + model in one file).  FIX (D)
    """
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Labeled data not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'towards' else -1)

    class_features = _build_classification_features(WINDOWS)
    logger.info(f"Classification features ({len(class_features)}): {class_features}")

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
        logger.warning(
            f"Only {n_unique_sessions} session(s) found — need ≥ 2 for CV. Skipping."
        )
    else:
        cv_run = True
        gkf    = GroupKFold(n_splits=n_folds)

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X_class, y_class, groups=sessions)
        ):
            X_tr = X_class.iloc[train_idx]
            y_tr = y_class.iloc[train_idx]
            X_te = X_class.iloc[test_idx]
            y_te = y_class.iloc[test_idx]

            X_tr_towards = X_tr[y_tr == 1]
            if len(X_tr_towards) < 5:
                logger.warning(f"  Fold {fold + 1}: < 5 'towards' samples — skipping.")
                continue

            # FIX (D): use Pipeline for CV reference model too
            cv_pipe = _make_occ_pipeline(OneClassSVM(kernel='rbf', nu=0.05))
            cv_pipe.fit(X_tr_towards)
            y_cv_pred = cv_pipe.predict(X_te)

            cv_f1.append(  f1_score(   y_te, y_cv_pred, pos_label=1, zero_division=0))
            cv_prec.append(precision_score(y_te, y_cv_pred, pos_label=1, zero_division=0))
            cv_rec.append( recall_score(   y_te, y_cv_pred, pos_label=1, zero_division=0))
            logger.info(
                f"  Fold {fold + 1} | sessions held out: "
                f"{np.unique(sessions[test_idx]).tolist()} | "
                f"F1={cv_f1[-1]:.4f}"
            )

    # -----------------------------------------------------------------------
    # STAGE 1 — OCC ALGORITHM SWEEP
    # -----------------------------------------------------------------------
    logger.info("--- Stage 1: OCC Algorithm Sweep ---")

    # Build a test set that contains all classes (stratified by binary_label)
    from sklearn.model_selection import train_test_split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, stratify=y_class, random_state=42
    )
    X_train_towards = X_train_c[y_train_c == 1]

    logger.info(
        f"  Train 'towards': {len(X_train_towards)} | "
        f"Test total: {len(X_test_c)} "
        f"(towards={( y_test_c==1).sum()}, not={( y_test_c==-1).sum()})"
    )

    # FIX (D): every classifier is wrapped in a Pipeline
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

        # Save individual pipeline artifact (scaler + model bundled)
        clean_name = (
            name.lower()
            .replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        )
        joblib.dump(pipe, f'models/pipeline_{clean_name}.joblib')

    detailed_df = pd.DataFrame(detailed_results)

    # Naive baseline
    logger.info("  Evaluating naive velocity baseline...")
    naive_result = _evaluate_naive_baseline(df, X_test_c.index, y_test_c)

    full_comparison_df = pd.concat(
        [pd.DataFrame([naive_result]), detailed_df],
        ignore_index=True
    )
    full_comparison_df.to_csv(
        os.path.join(REPORTS_DIR, 'detailed_algorithm_performance.csv'), index=False
    )

    # Best model — confusion matrix
    best_idx       = detailed_df['F1-Score'].idxmax()
    best_algo_name = detailed_df.iloc[best_idx]['Algorithm']
    best_f1        = detailed_df.iloc[best_idx]['F1-Score']
    logger.info(f"Best algorithm: {best_algo_name} (F1={best_f1:.4f})")

    best_pipe    = occ_pipelines[best_algo_name]
    y_pred_best  = best_pipe.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred_best, labels=[1, -1])

    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(['Actual: towards', 'Actual: not_towards'], name=''),
        columns=pd.Index(['Pred: towards', 'Pred: not_towards'], name=''),
    )

    TP, FP, FN, TN = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
    print("\n" + "=" * 60)
    print(f"CONFUSION MATRIX — {best_algo_name}")
    print("=" * 60)
    print(cm_df.to_string())
    print(f"\n  TP (caught approaches)   : {TP}")
    print(f"  FP (false alarms)        : {FP}  ← nuisance but safe")
    print(f"  FN (missed approaches)   : {FN}  ← safety risk, minimise this")
    print(f"  TN (correctly quiet)     : {TN}")
    print("=" * 60)

    cm_df.to_csv(os.path.join(REPORTS_DIR, 'confusion_matrix.csv'))

    # FIX (D): save Pipeline as the production motion detection artifact
    joblib.dump(best_pipe, 'models/motion_detection_model.joblib')
    logger.info(f"Production motion pipeline saved: {best_algo_name}")

    # -----------------------------------------------------------------------
    # STAGE 2 — TTC REGRESSION  (FIX B3: session-aware split)
    # -----------------------------------------------------------------------
    logger.info("\n--- Stage 2: TTC Regression (session-aware split) ---")

    regression_df = df[df['label'] == 'towards'].dropna(subset=['ttc']).copy()
    regression_df = regression_df[np.isfinite(regression_df['ttc'])]

    if regression_df.empty:
        logger.error("No valid TTC data for regression. Check label_generator output.")
        return

    reg_features = [
        'velocity', 'acceleration', 'echo_index',
        'Peak Frequency', 'Spectral Centroid', 'Trend_25',
    ]
    X_reg     = regression_df[reg_features]
    y_reg     = regression_df['ttc']
    reg_sessions = regression_df['session_id'].values

    # FIX (B3): GroupShuffleSplit keeps entire sessions in either train or test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_r, test_idx_r = next(gss.split(X_reg, y_reg, groups=reg_sessions))

    X_train_r = X_reg.iloc[train_idx_r]
    X_test_r  = X_reg.iloc[test_idx_r]
    y_train_r = y_reg.iloc[train_idx_r]
    y_test_r  = y_reg.iloc[test_idx_r]

    logger.info(
        f"  TTC regression split — train: {len(X_train_r)}, test: {len(X_test_r)} "
        f"(sessions in test: {np.unique(reg_sessions[test_idx_r]).tolist()})"
    )

    # FIX (D): regression models also wrapped in Pipelines
    reg_pipelines = {
        "Linear Regression (Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression()),
        ]),
        "SVR (RBF Kernel)": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  SVR(kernel='rbf', C=20.0, epsilon=0.01)),
        ]),
    }

    reg_results = []
    for name, pipe in reg_pipelines.items():
        logger.info(f"  Fitting {name}...")
        pipe.fit(X_train_r, y_train_r)
        y_pred_r = pipe.predict(X_test_r)

        mae  = mean_absolute_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2   = r2_score(y_test_r, y_pred_r)

        reg_results.append({"Model": name, "MAE (sec)": mae, "RMSE (sec)": rmse, "R2": r2})

        suffix = "linear" if "Linear" in name else "svr"
        joblib.dump(pipe, f'models/pipeline_ttc_{suffix}.joblib')

    reg_df = pd.DataFrame(reg_results)

    # -----------------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------------
    naive_f1_val = naive_result['F1-Score']
    margin       = best_f1 - naive_f1_val

    print("\n" + "=" * 60)
    print("STAGE 0: SESSION-AWARE CROSS-VALIDATION")
    if cv_run and cv_f1:
        print(f"  Reference model: One-Class SVM (nu=0.05)")
        print(f"  F1:        {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
        print(f"  Precision: {np.mean(cv_prec):.4f} ± {np.std(cv_prec):.4f}")
        print(f"  Recall:    {np.mean(cv_rec):.4f} ± {np.std(cv_rec):.4f}")
    else:
        print("  Not enough sessions to run cross-validation.")

    print("\nSTAGE 1: MOTION DETECTION — ALGORITHM SWEEP vs NAIVE BASELINE")
    print(full_comparison_df[['Algorithm', 'F1-Score', 'Accuracy', 'Precision', 'Recall']]
          .to_string(index=False))
    print(f"\n  Naive Baseline F1  : {naive_f1_val:.4f}")
    print(f"  Best ML model F1   : {best_f1:.4f}  ({best_algo_name})")
    print(f"  ML improvement     : +{margin:.4f}  "
          f"({'significant' if margin > 0.05 else 'marginal — reconsider model'})")

    print("\nSTAGE 2: TTC PREDICTION PERFORMANCE  (session-aware test split)")
    print(reg_df.to_string(index=False))
    print("=" * 60)

    best_ttc_idx  = reg_df['MAE (sec)'].idxmin()
    best_ttc_name = reg_results[best_ttc_idx]['Model']
    suffix        = "linear" if "Linear" in best_ttc_name else "svr"
    joblib.dump(reg_pipelines[best_ttc_name], 'models/ttc_prediction_model.joblib')
    logger.info(
        f"Production TTC pipeline: {best_ttc_name} "
        f"(MAE={reg_df.iloc[best_ttc_idx]['MAE (sec)']:.4f}s)"
    )


if __name__ == "__main__":
    train_master_pipeline()
