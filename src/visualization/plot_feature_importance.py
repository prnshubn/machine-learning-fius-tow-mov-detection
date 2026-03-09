"""
This script visualizes the most important features used by the trained 
motion detection model to make its decisions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

def plot_feature_importance():
    """
    Loads the trained model and plots the importance of each feature.
    """
    model_path = 'models/motion_detection_model.joblib'
    scaler_path = 'models/motion_scaler.joblib'
    # We use the final_labeled_data to get the feature names
    data_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"Error: Model or dataset not found. Please run training first.")
        return

    # Load the model and data
    print(f"Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Define the features that were used in training (MUST match src/models/03_train_and_compare.py)
    cols_to_drop = ['label', 'session_id', 'timestamp', 'velocity', 'acceleration']
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    feature_names = X.columns

    # Check if the model has feature_importances_ (Random Forest, Gradient Boosting)
    if not hasattr(model, 'feature_importances_'):
        print(f"The best model ({type(model).__name__}) does not support direct feature importance visualization.")
        print("This often happens with SVM or KNN. Skipping plot.")
        return

    # Create a DataFrame for visualization
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Most Useful Features for Motion Detection')
    plt.xlabel('Importance Score (0 to 1)')
    plt.ylabel('Sensor Feature')
    plt.tight_layout()
    
    # Save the plot
    output_path = 'reports/figures/feature_importance.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature importance plot saved to '{output_path}'")

if __name__ == "__main__":
    plot_feature_importance()
