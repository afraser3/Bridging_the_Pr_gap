"""
This script plots snapshots of dynamics in a 2D slice.

Usage:
    plot_xz_slices_hydro.py [options]

Options:
    --root_dir=<str>     Path to root directory containing data_dir [default: .]
    --data_dir=<str>     Name of data handler directory [default: slices]
    --out_name=<str>     Name of figure output directory & base name of saved figures [default: snapshots_xz]
    --start_fig=<int>    Number of first figure file [default: 1]
    --start_file=<int>   Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<float>    Total number of files to plot
    --dpi=<int>          Image pixel density [default: 200]

    --col_inch=<float>   Number of inches / column [default: 4]
    --row_inch=<float>   Number of inches / row [default: 4]
"""

from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None:
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)
# plotter = SlicePlotter(root_dir, sub_dir=data_dir, out_name=out_name, start_file=start_file, num_files=n_files)
plotter_kwargs = { 'col_inch' : float(args['--col_inch']), 'row_inch' : float(args['--row_inch']) }

if float(args['--row_inch']) >= 8:
    plotter.setup_grid(num_rows=1, num_cols=6, **plotter_kwargs)
else:
    plotter.setup_grid(num_rows=2, num_cols=3, **plotter_kwargs)

colors = 3*['PuOr_r'] + ['PRGn'] + ['PiYG_r'] + ['RdBu_r']
for field, cmap in zip(['u', 'v', 'w', 'C', 'T', 'b'], colors):
    if field=='b':
        plotter.add_colormesh(field + '_ywall', x_basis='x', y_basis='z', remove_x_mean=True, transpose=False, cmap=cmap)
    else:
        plotter.add_colormesh(field + '_ywall', x_basis='x', y_basis='z', transpose=False, cmap=cmap)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
