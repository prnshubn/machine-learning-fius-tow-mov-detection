"""
This script plots the distribution of labels in the final labeled dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_label_distribution():
    """
    Loads the final labeled dataset and plots the distribution of labels.
    """
    # --- File Path ---
    data_path = 'data/processed/final_labeled_data.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: The dataset '{data_path}' was not found.")
        print("Please run 'python3 src/data/label_generator.py' first to create it.")
        return

    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df, order=df['label'].value_counts().index)
    plt.title('Distribution of Motion Labels')
    plt.xlabel('Motion Label')
    plt.ylabel('Count')
    
    # Save the plot
    output_path = 'reports/figures/label_distribution.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Label distribution plot saved to '{output_path}'")

if __name__ == "__main__":
    plot_label_distribution()
