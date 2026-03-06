Code used for the paper *Bridging the Prandtl number gap: 3D simulations of thermohaline convection in astrophysical regimes*,
A.E. Fraser, currently under consideration for publication in Astrophysical Journal Letters.

Simulations are run using `hydro_DDC_IVP.py` (`hydro_DDC_IVP_LPN.py` for Pr = 0 case) as demonstrated in the script
`pleiades_hydro_DDC.pbs`, which also demonstrates how to merge outputs with `merge_tasks.py` and then plot snapshots
with `plot_xz_slices_hydro.py`. Scripts used in for the figures in the manuscript are `R0_tau_diagram.py`, 
`plot_w_ywall_comparison.py`, and `check_BGS13.py`.

Simulations rely on v2 of the [Dedalus pseudospectral solver](https://dedalus-project.org/). 
The snapshot plotting scripts use Evan Anders' [plotpal](https://github.com/evanhanders/plotpal) package; unfortunately
I think I'm using an outdated version, so users will have to either install an older version (ca 2024 or so?) or instead
simply replace my snapshot plotting scripts with something like [this](https://github.com/DedalusProject/dedalus/blob/v2_master/examples/ivp/3d_rayleigh_benard/plot_slices.py) 
example from the Dedalus v2 RBC example, suitably modified for this problem.