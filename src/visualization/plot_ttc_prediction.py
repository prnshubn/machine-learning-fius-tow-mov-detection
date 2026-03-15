"""
Visualization: TTC Prediction — Predicted vs Actual Scatter
Reads: data/processed/final_labeled_data.csv  +  models/ttc_prediction_model.joblib
Saves: reports/figures/ttc_prediction_scatter.png

Place this file in: src/visualization/plot_ttc_prediction.py
Run manually:
    python3 src/visualization/plot_ttc_prediction.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_DIR  = 'reports/figures'
DATA_PATH   = 'data/processed/final_labeled_data.csv'
MODEL_PATH  = 'models/ttc_prediction_model.joblib'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'ttc_prediction_scatter.png')

# Must match model_trainer.py Stage 2
REG_FEATURES = [
    'velocity', 'acceleration', 'echo_index',
    'Peak Frequency', 'Spectral Centroid', 'Trend_25',
]


def plot_ttc_prediction():
    for path in (DATA_PATH, MODEL_PATH):
        if not os.path.exists(path):
            print(f"Error: '{path}' not found. Run the training pipeline first.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    reg_df = (df[df['label'] == 'towards']
              .dropna(subset=['ttc'])
              .pipe(lambda d: d[np.isfinite(d['ttc'])]))

    if reg_df.empty:
        print("No valid TTC rows found.")
        return

    X = reg_df[REG_FEATURES]
    y = reg_df['ttc'].values

    # Reproduce the session-aware test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=reg_df['session_id'].values))

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#FAFAFA')

    # --- Left: scatter ---
    ax = axes[0]
    ax.set_facecolor('#FAFAFA')

    lim_min = max(0, min(y_test.min(), y_pred.min()) - 0.2)
    lim_max = max(y_test.max(), y_pred.max()) + 0.2

    ax.scatter(y_test, y_pred, alpha=0.55, s=28,
               color='#1D9E75', edgecolors='#085041', linewidths=0.4, zorder=3)

    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='#E24B4A', linewidth=1.4, linestyle='--',
            label='Perfect prediction', zorder=4)

    # ±0.5 s band
    ax.fill_between([lim_min, lim_max],
                    [lim_min - 0.5, lim_max - 0.5],
                    [lim_min + 0.5, lim_max + 0.5],
                    alpha=0.12, color='#1D9E75', label='±0.5 s band')

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Actual TTC (s)', fontsize=12)
    ax.set_ylabel('Predicted TTC (s)', fontsize=12)
    ax.set_title('TTC: Predicted vs Actual', fontsize=13, fontweight='500')

    ax.text(0.04, 0.96,
            f'MAE = {mae:.3f} s\nR²  = {r2:.3f}',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#9FE1CB', linewidth=0.8))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Right: residuals histogram ---
    ax2 = axes[1]
    ax2.set_facecolor('#FAFAFA')

    residuals = y_pred - y_test
    ax2.hist(residuals, bins=30, color='#1D9E75', edgecolor='#085041',
             linewidth=0.4, alpha=0.80)
    ax2.axvline(0, color='#E24B4A', linewidth=1.4, linestyle='--',
                label='Zero error')
    ax2.axvline(residuals.mean(), color='#BA7517', linewidth=1.2,
                linestyle=':', label=f'Mean residual ({residuals.mean():.3f} s)')

    ax2.set_xlabel('Residual: Predicted − Actual (s)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('TTC Residual Distribution', fontsize=13, fontweight='500')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    fig.suptitle('Time-to-Collision Regression Performance', fontsize=14,
                 fontweight='500', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TTC prediction scatter saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    plot_ttc_prediction()
