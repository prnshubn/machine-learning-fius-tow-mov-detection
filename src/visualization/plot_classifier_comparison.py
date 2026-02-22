"""
This script plots the classifier comparison results from the saved CSV file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_classifier_comparison():
    """
    Loads the classifier comparison results and generates a bar plot.
    """
    # --- File Path ---
    results_path = 'reports/classifier_comparison_results.csv'
    
    if not os.path.exists(results_path):
        print(f"Error: The results file '{results_path}' was not found.")
        print("Please run the training script first to create it.")
        return

    print(f"Loading results from '{results_path}'...")
    results_df = pd.read_csv(results_path, index_col=0)

    # --- Plotting ---
    # Update to match the new column names in 03_train_and_compare.py
    cols_to_plot = ['Accuracy', 'F1-approaching', 'F1-receding', 'F1-stationary']
    # Only plot columns that exist in the dataframe
    cols_to_plot = [col for col in cols_to_plot if col in results_df.columns]
    
    results_df_plot = results_df[cols_to_plot]
    
    ax = results_df_plot.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(0, 1.1) # Set limit slightly higher than 1.0 for legend
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    output_path = 'reports/figures/classifier_comparison.png'
    os.makedirs(os.path.dirname(output_dir := os.path.dirname(output_path)), exist_ok=True)
    plt.savefig(output_path)
    print(f"Classifier comparison plot saved to '{output_path}'")

if __name__ == "__main__":
    plot_classifier_comparison()
