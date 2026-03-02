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

plt.rcParams.update({"text.usetex": True})

# =============================================================================
# USER CONFIGURATION: Specify the three runs here
# =============================================================================
# Run for the left panel (aspect ratio 4, tall simulation)
RUN_LEFT = "eps0p1_tau5e-7_Pr1e-6_2z"
Ra_left = 1.1

# Run for the top-right panel (aspect ratio 2)
RUN_TOP_RIGHT = "eps10_tau5e-7_Pr1e-6_hres"
Ra_top_right = 11

# Run for the bottom-right panel (aspect ratio 2)
# RUN_BOTTOM_RIGHT = "eps100_tau5e-7_Pr1e-6_vvhres"
RUN_BOTTOM_RIGHT = "eps300_tau5e-7_Pr1e-6_vvhresx1p5"
Ra_bottom_right = 301

tau = 5e-7

# Base directory for runs
RUNS_DIR = Path(__file__).parent / "runs"

# Time snapshot index to plot (-1 for last snapshot in file)
snapshots = {
    'left': -1,
    'top_right': -1,
    'bottom_right': 75,
}
# =============================================================================


def eval_finger(R0, Pr, tau, rescale=False):
    # Evaluate the growth rate and horizontal wavenumber of fastest-growing
    # elevator mode. This is a slight modification to some very helpful code
    # originally written by Rich Townsend

    a3 = Pr * (1 - R0) + tau - R0
    a2 = -2 * (R0 - 1) * (Pr + tau + Pr * tau)
    a1 = Pr + tau - 4 * Pr * (R0 - 1) * tau - (1 + Pr) * R0 * tau ** 2
    a0 = -2 * Pr * tau * (R0 * tau - 1)

    p = a1 * a2 / (6 * a3 ** 2) - a0 / (2 * a3) - (a2 / (3 * a3)) ** 3
    q = (a2 / (3 * a3)) ** 2 - a1 / (3 * a3)

    tlam = 2 * np.sqrt(q) * np.cos(np.arccos(p / np.sqrt(q) ** 3) / 3) - a2 / (3 * a3)

    l_f = (Pr * (1 + tlam - R0 * (tau + tlam)) / (R0 * (1 + tlam) * (Pr + tlam) * (tau + tlam))) ** 0.25

    lam_f = l_f ** 2 * tlam

    if rescale:
        # divide by tau because the above equations are in units of kappa_T, but this project works in kappa_C units
        return lam_f/tau, l_f
    else:
        return lam_f, l_f


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
        print(np.shape(f['tasks']['w_ywall']))
        w_ywall = f['tasks']['w_ywall'][snapshot_idx, :, 0, :]

        # Get the coordinates
        x = f['scales']['x']['1.0'][:]
        z = f['scales']['z']['1.0'][:]

        # Get simulation time for the title
        sim_time = f['scales']['sim_time'][snapshot_idx]
        print(sim_time)

    return x, z, w_ywall, sim_time


def main():
    # Define runs
    runs = {
        'left': RUNS_DIR / RUN_LEFT,
        'top_right': RUNS_DIR / RUN_TOP_RIGHT,
        'bottom_right': RUNS_DIR / RUN_BOTTOM_RIGHT,
    }
    Ras = {
        'left': Ra_left,
        'top_right': Ra_top_right,
        'bottom_right': Ra_bottom_right,
    }

    # Load data from each run
    data = {}
    for key, run_dir in runs.items():
        filepath = get_latest_slices_file(run_dir)
        x, z, w, sim_time = load_w_ywall(filepath, snapshots[key])
        lam_f, l_f = eval_finger(1/(Ras[key]*tau), tau*2, tau, rescale=True)
        L_opt = 2*np.pi/l_f
        x /= L_opt
        z /= L_opt

        data[key] = {'x': x, 'z': z, 'w': w, 'sim_time': sim_time, 'run_name': run_dir.name}
        print(f"Loaded {key}: {run_dir.name}, t={sim_time:.4f}, shape={w.shape}")

    # Create figure with GridSpec
    # Left panel takes full height, right panels split vertically
    scale = 0.7
    fig = plt.figure(figsize=(6.5*scale, 8*scale))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  left=0.08, right=0.92, bottom=0.08, top=0.92,
                  wspace=0.05, hspace=0.15)

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

        pcm = ax.pcolormesh(X, Z, d['w'], cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        ax.set_aspect('equal')

        # Add left-adjusted subfigure labels
        label = {'left': r'(a) ~$\mathcal{R} = 1.1$', 'top_right': r'(b) ~$\mathcal{R} = 11$', 'bottom_right': r'(c) ~$\mathcal{R} = 301$'}[key]
        ax.set_title(label, loc='left', fontweight='bold')

        # Set axis labels conditionally
        if key != 'top_right':
            ax.set_xlabel(r'$\tilde{x}/(2 \pi/\tilde{k}_{opt})$', fontsize='large')
        if key == 'left':
            ax.set_ylabel(r'$\tilde{z}/(2 \pi/\tilde{k}_{opt})$', fontsize='large')
            ax.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, 32])
        else:
            ax.set_yticks([0, 4, 8, 12, 16])
        if key == 'top_right':
            ax.xaxis.set_tick_params(which='both', labelbottom=False)
        ax.set_xticks([0, 4, 8])

        # Individual colorbar for each panel
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
        # cbar.set_label(r'$\tilde{u}_z$')

    plt.savefig(RUNS_DIR.parent / 'figures/w_ywall_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(RUNS_DIR.parent / 'figures/w_ywall_comparison.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Saved figure to {RUNS_DIR.parent / 'figures/w_ywall_comparison.png'}")


if __name__ == "__main__":
    main()
