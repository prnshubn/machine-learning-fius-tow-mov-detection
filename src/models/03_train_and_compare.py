
"""
This script compares the performance of several well-known classifiers,
selects the best one, and saves it for future use.

Main functionalities:
- Load labeled feature dataset
- Train multiple classifiers (Random Forest, Logistic Regression, SVM, etc.)
- Compare their performance
- Save the best performing model for later use
"""

# Import required libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report  # For evaluation
from sklearn.preprocessing import StandardScaler # For feature scaling
import os  # For file operations
import sys
import warnings  # To suppress warnings
import joblib  # For saving models

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_and_compare():
    """
    Loads the dataset, trains multiple classifiers, saves the best one, and prints a comparison.
    """
    # --- Load the Dataset ---
    feature_dataset_path = 'data/processed/final_labeled_data.csv'  # Path to labeled data
    
    # Check if the feature dataset exists
    if not os.path.exists(feature_dataset_path):
        print(f"Error: The feature dataset '{feature_dataset_path}' was not found.")
        print("Please run 'python3 src/data/02_refine_labels.py' first to create it.")
        sys.exit(1)

    # Load the labeled feature dataset
    print(f"Loading feature dataset from '{feature_dataset_path}'...")
    df = pd.read_csv(feature_dataset_path)

    # Note: We now KEEP the 'stationary' class to make the model more robust.
    
    # Check if there is enough data to train
    if df.shape[0] < 10:
        print("Not enough data to train the models. Aborting.")
        sys.exit(1)

    # --- Prepare Data for Modeling ---
    # CRITICAL FIX: To avoid label leakage, we MUST remove 'velocity' and 'acceleration'
    # as they were derived from the same distance change used to create the labels.
    # We also remove other non-feature columns.
    cols_to_drop = ['label', 'session_id', 'timestamp', 'velocity', 'acceleration']
    
    # Optional: You might also want to remove 'distance' if you want the model 
    # to rely ONLY on the Doppler shift (FFT features). 
    # For now, we'll keep distance as it can help distinguish distance-based signal attenuation.
    # cols_to_drop.append('distance') 
        
    X = df.drop(cols_to_drop, axis=1, errors='ignore')  # Features
    y = df['label']                    # Target labels
    
    print(f"Training on features: {list(X.columns)}")
    
    # --- Split Data into Training and Testing Sets ---
    # Use stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- Feature Scaling ---
    # CRITICAL: Scale features for models like SVM, KNN, and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler! Prediction scripts need it.
    scaler_path = 'models/motion_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to '{scaler_path}'")

    # --- Define Classifiers ---
    # Dictionary of classifiers to compare
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine": SVC(random_state=42, probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # --- Train and Evaluate Each Classifier ---
    for name, clf in classifiers.items():
        print(f"--- Training {name} ---")
        # Use scaled data for training
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Collect basic metrics for comparison table
        metrics = {"Accuracy": accuracy}
        for label in ['approaching', 'receding', 'stationary']:
            if label in report:
                metrics[f"F1-{label}"] = report[label]['f1-score']
        
        results[name] = metrics
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_name = name

        print(f"--- Evaluation for {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("---------------------------------")

    # --- Save Summary Table ---
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Classifier Model' # Add name to the first column
    print("--- Classifier Performance Comparison ---")
    print(results_df)
    
    # Save results to a CSV file
    output_path = 'reports/classifier_comparison_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path)
    print(f"Comparison results saved to '{output_path}'")

    # --- Save the Best Model ---
    model_path = 'models/motion_detection_model.joblib'
    print(f"--- Saving the best model ({best_model_name}) to '{model_path}' ---")
    joblib.dump(best_model, model_path)
    print("Best model saved successfully.")

if __name__ == "__main__":
    train_and_compare()
