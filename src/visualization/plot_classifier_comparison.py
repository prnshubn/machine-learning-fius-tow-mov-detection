"""
This script plots the classifier comparison results from the detailed OCC performance report.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_classifier_comparison():
    """
    Loads the detailed algorithm performance report and generates a bar plot.
    """
    # --- File Path ---
    # Updated to match the new reporting format from model_trainer.py
    results_path = 'reports/detailed_algorithm_performance.csv'
    
    if not os.path.exists(results_path):
        print(f"Error: The results file '{results_path}' was not found.")
        print("Please run 'bash run_pipeline.sh' to generate the performance data.")
        return

    print(f"Loading results from '{results_path}'...")
    # Read CSV and use 'Algorithm' as the index for plotting
    results_df = pd.read_csv(results_path)
    results_df.set_index('Algorithm', inplace=True)

    # --- Plotting ---
    # The new columns are: F1-Score, Accuracy, Precision, Recall
    cols_to_plot = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
    
    # Filter for columns that actually exist in the CSV
    available_cols = [col for col in cols_to_plot if col in results_df.columns]
    
    if not available_cols:
        print("Error: No performance metrics found in the results file.")
        return

    ax = results_df[available_cols].plot(kind='bar', figsize=(12, 7), width=0.8)
    
    plt.title('One-Class Classification Performance Comparison', fontsize=14)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylim(0, 1.1) 
    plt.xticks(rotation=45, ha='right') # Rotate for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    
    # Save the plot
    output_path = 'reports/figures/classifier_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Comparison plot showing {len(results_df)} algorithms saved to '{output_path}'")

if __name__ == "__main__":
    plot_classifier_comparison()
