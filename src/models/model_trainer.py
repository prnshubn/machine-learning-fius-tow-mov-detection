"""
Model Training & Comparison Module: Evaluates multiple One-Class algorithms.
Uses QUAD-SCALE temporal context (5, 10, 25, 50 frames) for pattern optimization.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Import shared detector logic
from src.models.detectors import AutoencoderDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate_all():
    """
    Trains and tunes OCC algorithms using Quad-Scale features.
    """
    feature_dataset_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(feature_dataset_path):
        logger.error(f"Feature dataset not found.")
        sys.exit(1)

    df = pd.read_csv(feature_dataset_path)
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'towards' else -1)
    
    # --- UPDATED: Quad-Scale Features ---
    cols_to_use = ['echo_index', 'Peak Frequency', 'Spectral Centroid']
    for w in [5, 10, 25, 50]:
        cols_to_use.extend([f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}'])
    
    # Filter for columns that actually exist
    cols_to_use = [c for c in cols_to_use if c in df.columns]
    X = df[cols_to_use]
    y = df['binary_label']
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_towards = X_train_all[y_train_all == 1]
    logger.info(f"Training on {len(X_train_towards)} Quad-Scale 'towards' samples.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_towards)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/motion_scaler.joblib')

    # Hyperparameter configurations
    classifiers_to_tune = [
        ("One-Class SVM (nu=0.01)", OneClassSVM(kernel='rbf', nu=0.01)),
        ("One-Class SVM (nu=0.05)", OneClassSVM(kernel='rbf', nu=0.05)),
        ("Isolation Forest (100 est)", IsolationForest(n_estimators=100, contamination=0.05, random_state=42)),
        ("Isolation Forest (200 est)", IsolationForest(n_estimators=200, contamination=0.03, random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)),
        ("Autoencoder (Narrow)", AutoencoderDetector(hidden_layer_sizes=(4, 2, 4), threshold_quantile=0.97)),
        ("Autoencoder (Wide)", AutoencoderDetector(hidden_layer_sizes=(8, 4, 8), threshold_quantile=0.95))
    ]

    results = []

    for name, clf in classifiers_to_tune:
        logger.info(f"Tuning Quad-Scale: {name}")
        clf.fit(X_train_scaled)
        y_pred = clf.predict(X_test_scaled)
        
        f1 = f1_score(y_test, y_pred, pos_label=1)
        results.append({
            "Algorithm": name, "F1-Score": f1, "Accuracy": accuracy_score(y_test, y_pred), 
            "Precision": precision_score(y_test, y_pred, pos_label=1), 
            "Recall": recall_score(y_test, y_pred, pos_label=1)
        })
        
        clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        joblib.dump(clf, f"models/model_{clean_name}.joblib")

    report_df = pd.DataFrame(results)
    logger.info("\n" + "="*60 + "\nQUAD-SCALE ALGORITHM PERFORMANCE\n" + "="*60)
    print(report_df.sort_values(by='F1-Score', ascending=False).to_string(index=False))
    
    report_df.to_csv('reports/detailed_algorithm_performance.csv', index=False)

    best_row = report_df.loc[report_df['F1-Score'].idxmax()]
    best_name = best_row['Algorithm']
    
    # Save the absolute best
    for name, clf in classifiers_to_tune:
        if name == best_name:
            joblib.dump(clf, 'models/motion_detection_model.joblib')
            break
            
    logger.info(f"Best Quad-Scale model: {best_name} (F1: {best_row['F1-Score']:.4f})")

if __name__ == "__main__":
    train_and_evaluate_all()
