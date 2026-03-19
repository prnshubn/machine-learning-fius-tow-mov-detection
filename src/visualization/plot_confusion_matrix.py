"""
Visualization: Confusion Matrix Heatmap
Reads: reports/confusion_matrix.csv
Saves: reports/figures/confusion_matrix.png

Place this file in: src/visualization/plot_confusion_matrix.py
Run automatically via run_pipeline.sh (Step 5), or manually:
    python3 src/visualization/plot_confusion_matrix.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR   = 'reports/figures'
INPUT_PATH   = 'reports/confusion_matrix.csv'
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')


def plot_confusion_matrix():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: '{INPUT_PATH}' not found. Run the training pipeline first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ load
    cm_df = pd.read_csv(INPUT_PATH, index_col=0)
    cm    = cm_df.values.astype(int)           # [[TP, FN], [FP, TN]]

    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    total  = cm.sum()

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Custom colour map: deep teal for correct, muted coral for errors
    cell_colors = np.array([
        ['#1D9E75', '#F0997B'],   # TP (good), FN (dangerous miss)
        ['#FAC775', '#1D9E75'],   # FP (nuisance), TN (good)
    ])
    text_colors = [
        ['white',  'white'],
        ['#412402', 'white'],
    ]

    # Draw coloured cells
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (c, 1 - r), 1, 1,
                color=cell_colors[r, c], zorder=1
            ))

    # Large count numbers
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            pct = 100.0 * val / total
            ax.text(
                c + 0.5, 1.5 - r, str(val),
                ha='center', va='center',
                fontsize=36, fontweight='bold',
                color=text_colors[r][c], zorder=2
            )
            ax.text(
                c + 0.5, 1.5 - r - 0.28, f'{pct:.1f}%',
                ha='center', va='center',
                fontsize=13, color=text_colors[r][c], alpha=0.85, zorder=2
            )

    # Cell annotation labels (corner, small)
    labels = [
        ['TP — caught approach', 'FN — missed approach ⚠'],
        ['FP — false alarm',     'TN — correctly quiet'],
    ]
    for r in range(2):
        for c in range(2):
            ax.text(
                c + 0.05, 1.97 - r, labels[r][c],
                ha='left', va='top',
                fontsize=9, color=text_colors[r][c], alpha=0.80, zorder=2
            )

    # Axes labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Predicted: towards', 'Predicted: not towards'],
                       fontsize=12)
    ax.set_yticklabels(['Actual: not towards', 'Actual: towards'],
                       fontsize=12)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Derived metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (TP + TN) / total

    metrics_text = (
        f'Precision: {precision:.3f}   Recall: {recall:.3f}   '
        f'F1: {f1:.3f}   Accuracy: {accuracy:.3f}'
    )
    fig.text(0.5, 0.04, metrics_text, ha='center', fontsize=11,
             color='#444441',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E1F5EE',
                       edgecolor='#1D9E75', linewidth=0.8))

    ax.set_title('Motion Detection — Confusion Matrix', fontsize=14,
                 fontweight='500', pad=16)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    plot_confusion_matrix()
