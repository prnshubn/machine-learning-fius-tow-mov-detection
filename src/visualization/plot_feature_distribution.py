"""
This script plots the distribution of a selected feature for each label.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_distribution():
    """
    Loads the feature dataset and plots the distribution of a feature for each label.
    """
    # --- File Path ---
    data_path = 'data/processed/features.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: The dataset '{data_path}' was not found.")
        print("Please run 'python3 src/data/01_build_features.py' first to create it.")
        return

    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='label', y='Peak Frequency', data=df)
    plt.title('Distribution of Peak Frequency for Each Label')
    plt.xlabel('Label')
    plt.ylabel('Peak Frequency (Hz)')
    plt.xticks(rotation=45)
    
    # Save the plot
    output_path = 'reports/figures/feature_distribution.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature distribution plot saved to '{output_path}'")

if __name__ == "__main__":
    plot_feature_distribution()
