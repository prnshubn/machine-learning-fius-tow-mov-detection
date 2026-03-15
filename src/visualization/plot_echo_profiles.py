"""
Visualization: Echo Profiles by Object Type
Reads: data/processed/features.csv
Saves: reports/figures/echo_profiles.png

Shows how echo_amplitude (signal strength) and echo_width (FWHM, surface
hardness proxy) differ across object types — metal plate vs cardboard vs human.
This is the key justification for including those two features in the classifier.

Place this file in: src/visualization/plot_echo_profiles.py
Run manually:
    python3 src/visualization/plot_echo_profiles.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR  = 'reports/figures'
DATA_PATH   = 'data/processed/features.csv'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'echo_profiles.png')

# Colour palette — one per object type (auto-assigned)
PALETTE = ['#1D9E75', '#D85A30', '#378ADD', '#D4537E', '#639922', '#BA7517']


def plot_echo_profiles():
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' not found. Run feature extraction first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Drop rows where no echo was detected (echo_index == -1 → amplitude == 0)
    df_echo = df[df['echo_amplitude'] > 0].copy()

    if df_echo.empty:
        print("No valid echo rows found in features.csv")
        return

    labels = sorted(df_echo['label'].unique())
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#FAFAFA')

    metrics = [
        ('echo_amplitude', 'Echo Amplitude (ADC counts)',
         'Echo Amplitude by Object Type',
         'Stronger echo = object closer or more reflective surface'),
        ('echo_width',     'Echo Width — FWHM (samples)',
         'Echo Width (FWHM) by Object Type',
         'Wider echo = softer / more diffuse surface (e.g. human vs metal)'),
    ]

    for ax, (col, ylabel, title, subtitle) in zip(axes, metrics):
        ax.set_facecolor('#FAFAFA')

        data_per_label = [df_echo.loc[df_echo['label'] == lbl, col].dropna().values
                          for lbl in labels]

        bp = ax.boxplot(
            data_per_label,
            patch_artist=True,
            notch=False,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker='o', markersize=3, alpha=0.35, linewidth=0),
        )

        for patch, lbl in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[lbl])
            patch.set_alpha(0.82)

        for flier, lbl in zip(bp['fliers'], labels):
            flier.set_markerfacecolor(colors[lbl])
            flier.set_markeredgecolor(colors[lbl])

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(
            [lbl.replace('_', ' ').title() for lbl in labels],
            fontsize=11,
            rotation=45,
            ha='right'
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='500', pad=20)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha='center', fontsize=9, color='#5F5E5A', style='italic')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

        # Median annotations
        for i, (data, lbl) in enumerate(zip(data_per_label, labels), start=1):
            if len(data):
                ax.text(i, np.median(data), f' {np.median(data):.0f}',
                        va='center', fontsize=9,
                        color=colors[lbl], fontweight='bold')

    legend_patches = [
        mpatches.Patch(facecolor=colors[lbl], label=lbl.replace('_', ' ').title())
        for lbl in labels
    ]
    fig.legend(handles=legend_patches, loc='upper center',
               ncol=len(labels), fontsize=10,
               bbox_to_anchor=(0.5, 1.04), frameon=False)

    fig.suptitle('Echo Shape Features — Material Discrimination', fontsize=14,
                 fontweight='500', y=1.09)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Echo profiles plot saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    plot_echo_profiles()
