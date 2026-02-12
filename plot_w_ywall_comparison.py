"""
Plot w_ywall slices from three different DDC runs.
Left: one run with aspect ratio 4 (tall)
Right: two runs with aspect ratio 2 (top-right and bottom-right)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob

# =============================================================================
# USER CONFIGURATION: Specify the three runs here
# =============================================================================
# Run for the left panel (aspect ratio 4, tall simulation)
RUN_LEFT = "eps0p1_tau5e-7_Pr1e-6_2z"

# Run for the top-right panel (aspect ratio 2)
RUN_TOP_RIGHT = "eps10_tau5e-7_Pr1e-6_hres"

# Run for the bottom-right panel (aspect ratio 2)
RUN_BOTTOM_RIGHT = "eps100_tau5e-7_Pr1e-6_vvhres"

# Base directory for runs
RUNS_DIR = Path(__file__).parent / "runs"

# Time snapshot index to plot (-1 for last snapshot in file)
SNAPSHOT_INDEX = -1
# =============================================================================


def get_latest_slices_file(run_dir):
    """Get the most recent slices file from a run directory."""
    slices_dir = run_dir / "slices"
    slices_files = sorted(glob.glob(str(slices_dir / "slices_s*.h5")))
    if not slices_files:
        raise FileNotFoundError(f"No slices files found in {slices_dir}")
    return slices_files[-1]


def load_w_ywall(filepath, snapshot_idx=-1):
    """Load w_ywall data and coordinates from an HDF5 slices file."""
    with h5py.File(filepath, 'r') as f:
        # Get the data - shape is (time, x, y=1, z)
        w_ywall = f['tasks']['w_ywall'][snapshot_idx, :, 0, :]

        # Get the coordinates
        x = f['scales']['x']['1.0'][:]
        z = f['scales']['z']['1.0'][:]

        # Get simulation time for the title
        sim_time = f['scales']['sim_time'][snapshot_idx]

    return x, z, w_ywall, sim_time


def main():
    # Define runs
    runs = {
        'left': RUNS_DIR / RUN_LEFT,
        'top_right': RUNS_DIR / RUN_TOP_RIGHT,
        'bottom_right': RUNS_DIR / RUN_BOTTOM_RIGHT,
    }

    # Load data from each run
    data = {}
    for key, run_dir in runs.items():
        filepath = get_latest_slices_file(run_dir)
        x, z, w, sim_time = load_w_ywall(filepath, SNAPSHOT_INDEX)
        data[key] = {'x': x, 'z': z, 'w': w, 'sim_time': sim_time, 'run_name': run_dir.name}
        print(f"Loaded {key}: {run_dir.name}, t={sim_time:.4f}, shape={w.shape}")

    # Create figure with GridSpec
    # Left panel takes full height, right panels split vertically
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  left=0.08, right=0.92, bottom=0.08, top=0.92,
                  wspace=0.02, hspace=0.15)

    # Left panel spans both rows
    ax_left = fig.add_subplot(gs[:, 0])
    # Right panels
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    axes = {
        'left': ax_left,
        'top_right': ax_top_right,
        'bottom_right': ax_bottom_right,
    }

    # Plot each panel with individual color scales
    for key, ax in axes.items():
        d = data[key]
        X, Z = np.meshgrid(d['x'], d['z'], indexing='ij')

        # Individual color limits for each panel
        vmax = np.max(np.abs(d['w']))
        vmin = -vmax

        pcm = ax.pcolormesh(X, Z, d['w'], cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_aspect('equal')

        # Add left-adjusted subfigure labels
        label = {'left': '(a)', 'top_right': '(b)', 'bottom_right': '(c)'}[key]
        ax.set_title(label, loc='left', fontweight='bold')

        # Set axis labels conditionally
        if key != 'top_right':
            ax.set_xlabel('x')
        if key == 'left':
            ax.set_ylabel('z')

        # Individual colorbar for each panel
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('w')

    plt.savefig(RUNS_DIR.parent / 'w_ywall_comparison.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"Saved figure to {RUNS_DIR.parent / 'w_ywall_comparison.png'}")


if __name__ == "__main__":
    main()
