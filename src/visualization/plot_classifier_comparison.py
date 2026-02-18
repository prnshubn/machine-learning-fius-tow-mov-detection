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
        print("Please run 'python3 src/models/05_compare_classifiers.py' first to create it.")
        return

    print(f"Loading results from '{results_path}'...")
    results_df = pd.read_csv(results_path, index_col=0)

    # --- Plotting ---
    results_df_plot = results_df[['Accuracy', 'F1-Score (approaching)', 'F1-Score (receding)']]
    results_df_plot.plot(kind='bar', figsize=(14, 8))
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_path = 'reports/figures/classifier_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Classifier comparison plot saved to '{output_path}'")

if __name__ == "__main__":
    plot_classifier_comparison()
