"""
Visualization: Feature Correlation Heatmap
Reads: data/processed/final_labeled_data.csv
Saves: reports/figures/feature_correlation.png

Shows Pearson correlation between the 17 classification features and the
binary motion label. Useful for feature selection discussion and for
explaining to the professor which signals carry the most discriminative power.

Place this file in: src/visualization/plot_feature_correlation.py
Run manually:
    python3 src/visualization/plot_feature_correlation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_DIR  = 'reports/figures'
DATA_PATH   = 'data/processed/final_labeled_data.csv'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'feature_correlation.png')

WINDOWS = [5, 10, 25, 50]

FEATURE_COLS = (
    ['echo_index', 'echo_amplitude', 'echo_width',
     'Peak Frequency', 'Spectral Centroid']
    + [f for w in WINDOWS for f in
       (f'Trend_{w}', f'Mean_{w}', f'Centroid_Trend_{w}')]
)

# Human-readable short names for the axis labels
SHORT_NAMES = {
    'echo_index':        'Echo index',
    'echo_amplitude':    'Echo amplitude',
    'echo_width':        'Echo width (FWHM)',
    'Peak Frequency':    'Peak freq.',
    'Spectral Centroid': 'Spectral centroid',
}
for w in WINDOWS:
    SHORT_NAMES[f'Trend_{w}']          = f'Trend {w}'
    SHORT_NAMES[f'Mean_{w}']           = f'Mean {w}'
    SHORT_NAMES[f'Centroid_Trend_{w}'] = f'Centroid trend {w}'


def _diverging_cmap():
    """Teal → white → coral diverging palette."""
    return mcolors.LinearSegmentedColormap.from_list(
        'teal_coral',
        ['#085041', '#1D9E75', '#E1F5EE', '#FAFAFA',
         '#FAECE7', '#D85A30', '#4A1B0C'],
        N=256
    )


def plot_feature_correlation():
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' not found. Run the labelling pipeline first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df['binary_label'] = (df['label'] == 'towards').astype(int)

    # Keep only columns that exist (guard against missing cols)
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        print("No feature columns found in the dataset.")
        return

    corr_df = df[available + ['binary_label']].fillna(0).corr()
    cols_plus_label = available + ['binary_label']
    corr_matrix = corr_df.loc[cols_plus_label, cols_plus_label].values

    tick_labels = [SHORT_NAMES.get(c, c) for c in available] + ['Motion label']

    n = len(tick_labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(9, n * 0.5)))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    cmap = _diverging_cmap()
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text_color = 'white' if abs(val) > 0.55 else '#2C2C2A'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=7.5, color=text_color)

    # Highlight the label column/row with a border
    label_idx = n - 1
    for idx in (label_idx,):
        ax.add_patch(plt.Rectangle(
            (idx - 0.5, -0.5), 1, n,
            linewidth=1.5, edgecolor='#BA7517', facecolor='none', zorder=4
        ))
        ax.add_patch(plt.Rectangle(
            (-0.5, idx - 0.5), n, 1,
            linewidth=1.5, edgecolor='#BA7517', facecolor='none', zorder=4
        ))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Pearson r', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        'Feature Correlation Matrix\n'
        '(amber border = correlation with motion label)',
        fontsize=13, fontweight='500', pad=12
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature correlation heatmap saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    plot_feature_correlation()
