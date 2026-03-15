"""
Visualization: Velocity Timeline with Motion Labels
Reads: data/processed/final_labeled_data.csv
Saves: reports/figures/velocity_timeline.png

Plots the Kalman-filtered velocity for each recording session with 'towards'
windows shaded green and TTC values annotated. This is the most intuitive
figure for explaining what the labelling pipeline actually does.

Place this file in: src/visualization/plot_velocity_timeline.py
Run manually:
    python3 src/visualization/plot_velocity_timeline.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR  = 'reports/figures'
DATA_PATH   = 'data/processed/final_labeled_data.csv'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'velocity_timeline.png')

TOWARDS_COLOR     = '#1D9E75'
NOT_TOWARDS_COLOR = '#F09595'
VELOCITY_COLOR    = '#185FA5'
TTC_COLOR         = '#BA7517'
THRESHOLD_COLOR   = '#E24B4A'
VELOCITY_THRESHOLD = -0.05     # must match label_generator.py


def _shade_motion_regions(ax, frame_indices, labels, y_min, y_max):
    """Fill contiguous 'towards' blocks with a translucent green band."""
    in_towards = False
    start = None
    for idx, lbl in zip(frame_indices, labels):
        if lbl == 'towards' and not in_towards:
            in_towards = True
            start = idx
        elif lbl != 'towards' and in_towards:
            ax.axvspan(start, idx, alpha=0.18, color=TOWARDS_COLOR, zorder=1)
            in_towards = False
    if in_towards:
        ax.axvspan(start, frame_indices[-1], alpha=0.18, color=TOWARDS_COLOR, zorder=1)


def plot_velocity_timeline():
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' not found. Run label generation first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    sessions = df['session_id'].unique()
    n = len(sessions)

    # Limit to a reasonable number of subplots if there are many sessions
    MAX_SESSIONS = 6
    if n > MAX_SESSIONS:
        print(f"  Showing first {MAX_SESSIONS} of {n} sessions.")
        sessions = sessions[:MAX_SESSIONS]
        n = MAX_SESSIONS

    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#FAFAFA')

    for ax, session in zip(axes, sessions):
        ax.set_facecolor('#FAFAFA')
        grp = df[df['session_id'] == session].reset_index(drop=True)
        frames = grp.index.values
        vel    = grp['velocity'].values

        # Shade 'towards' windows
        _shade_motion_regions(ax, frames, grp['label'].values,
                              vel.min() - 0.02, vel.max() + 0.02)

        # Velocity line
        ax.plot(frames, vel, color=VELOCITY_COLOR,
                linewidth=1.2, alpha=0.85, zorder=3, label='Kalman velocity')

        # Detection threshold
        ax.axhline(VELOCITY_THRESHOLD, color=THRESHOLD_COLOR,
                   linewidth=0.9, linestyle='--', alpha=0.7,
                   label=f'Threshold ({VELOCITY_THRESHOLD} m/s)')

        # TTC annotations (show only a few to avoid clutter)
        towards_rows = grp[(grp['label'] == 'towards') & grp['ttc'].notna()]
        shown = 0
        for _, row in towards_rows.iterrows():
            if shown >= 5:
                break
            ax.annotate(
                f"TTC≈{row['ttc']:.1f}s",
                xy=(row.name, row['velocity']),
                xytext=(0, 14), textcoords='offset points',
                fontsize=7.5, color=TTC_COLOR,
                arrowprops=dict(arrowstyle='->', color=TTC_COLOR,
                                lw=0.7, shrinkA=0, shrinkB=2),
                ha='center', zorder=5,
            )
            shown += 1

        ax.set_ylabel('Velocity (m/s)', fontsize=10)
        ax.set_xlabel('Frame index', fontsize=10)
        ax.set_title(
            f"Session: {session.replace('_', ' ').title()}",
            fontsize=12, fontweight='500'
        )
        ax.grid(True, linestyle='--', alpha=0.35, axis='y')
        ax.spines[['top', 'right']].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=TOWARDS_COLOR, alpha=0.35,
                       label='Towards region (labelled)'),
        plt.Line2D([0], [0], color=VELOCITY_COLOR,   linewidth=1.4,
                   label='Kalman velocity'),
        plt.Line2D([0], [0], color=THRESHOLD_COLOR,  linewidth=1.0,
                   linestyle='--', label='Detection threshold'),
        plt.Line2D([0], [0], color=TTC_COLOR,        linewidth=0,
                   marker='>', markersize=6, label='TTC annotation'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    fig.suptitle('Kalman Velocity Timeline with Motion Labels',
                 fontsize=14, fontweight='500', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Velocity timeline saved to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    plot_velocity_timeline()
