"""
This script plots the distance over time for a given raw data file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_distance_over_time(filepath):
    """
    Loads a raw data file and plots the distance over time.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return

    print(f"Loading data from '{filepath}'...")
    df = pd.read_csv(filepath, header=None)
    
    # Assuming the distance is in column 10
    distance = df.iloc[:, 10]

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    plt.plot(distance)
    plt.title(f'Distance Over Time for {os.path.basename(filepath)}')
    plt.xlabel('Measurement Index (Time)')
    plt.ylabel('Distance')
    
    # Save the plot
    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'distance_over_time_{os.path.basename(filepath).replace(".csv", ".png")}')
    plt.savefig(output_filename)
    print(f"Distance plot saved to '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot distance over time for a raw data file.')
    parser.add_argument('filepath', type=str, help='The path to the raw CSV data file.')
    args = parser.parse_args()
    plot_distance_over_time(args.filepath)
