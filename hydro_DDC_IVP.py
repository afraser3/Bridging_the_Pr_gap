"""
3D DDC model in Dedalus v2

To run in parallel with 2 processes, e.g., do:
    $ mpiexec -n 2 python3 hydro_DDC_IVP.py config_files/test.cfg

Optionally, you can specify a subdirectory within IVPs that you want it to save to.
So, the following will save to runs/subdir/test/ (might have to make sure 'subdir' exists, I forget)
    $ mpiexec -n 2 python3 hydro_DDC_IVP.py config_files/test.cfg subdir

Most of the parameters in the .cfg file are hopefully self-explanatory. But note that:
- `R0_is_epsilon` and `reduced_r` control whether the entry `R0` is interpreted as the density ratio R0, the reduced
  density ratio r, or as the supercriticality epsilon = 1/(R0 * tau) - 1
- kopt_Lscale controls whether the input Lx, Ly, Lz are interpreted in units of d (characteristic finger width) or of
  2pi/k_opt (the wavelength of the fastest-growing mode of instability).
- buoyancy_dtmax controls whether or not the max timestep size is set to a fraction (set by bdt_frac) of the buoyancy
  frequency or not.
"""

import numpy as np
from mpi4py import MPI

CW = MPI.COMM_WORLD
import time

from dedalus import public as de
from dedalus.extras import flow_tools

from pathlib import Path
import sys
from configparser import ConfigParser

import logging

logger = logging.getLogger(__name__)


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


def r_from_R(R, tau):
    return (R-1)/(-1 + 1/tau)


def R_from_r(r, tau):
    return r*(-1 + 1/tau) + 1


# Parse the .cfg filename passed to script
config_file = Path(sys.argv[1])
logger.info("Running with config file {}".format(str(config_file)))

# Parse settings from .cfg file and save to variables
runconfig = ConfigParser()
runconfig.read(str(config_file))
if len(sys.argv) == 2:
    # data dir to save to: if "config_files/run1.cfg" is input, then "../IVPs/run1/" becomes the output directory
    basedir = Path("runs") / config_file.stem
else:
    basedir = Path("runs") / sys.argv[2] / config_file.stem
if CW.rank == 0:  # only do this for the 0th MPI process
    if not basedir.exists():
        basedir.mkdir(parents=True)

# Parameters
params_in = runconfig['parameters']
kopt_Lscale = params_in.getboolean('kopt_Lscale', fallback=False)
tau = params_in.getfloat('tau', fallback=0.1)
Pr = params_in.getfloat('Pr', fallback=0.1)
Sc = Pr/tau
rescale = params_in.getboolean('rescale', fallback=False)
implicit_IGWs = params_in.getboolean('implicit_IGWs', fallback=True)
R0_is_epsilon = params_in.getboolean('R0_is_epsilon', fallback=False)  # the parameter named "R0" in the input file is actually supercriticality ("epsilon")
reduced_r = params_in.getboolean('reduced_r', fallback=False)  # the parameter named "R0" in the input file is actually r ("reduced density ratio")
if R0_is_epsilon and reduced_r:
    raise ValueError("R0_is_epsilon and reduced_r cannot both be True")
elif R0_is_epsilon:
    epsilon = params_in.getfloat('R0', fallback=0.1)
    Rayleigh = epsilon + 1  # see eqn 2.1 of Fraser et al 2025 (helical flows paper)
    R0 = 1/(Rayleigh*tau)
elif reduced_r:
    R0 = R_from_r(params_in.getfloat('R0', fallback=2), tau)
else:
    R0 = params_in.getfloat('R0', fallback=2)
Ra = 1/(R0*tau)
lambda_opt, k_opt = eval_finger(R0, Pr, tau, rescale)
wavelength_opt = 2*np.pi/k_opt
Nx = params_in.getint('Nx', fallback=64)
Ny = params_in.getint('Ny', fallback=64)
Nz = params_in.getint('Nz', fallback=64)
if kopt_Lscale:
    Lx = params_in.getfloat('Lx', fallback=8) * wavelength_opt
    Ly = params_in.getfloat('Ly', fallback=4) * wavelength_opt
    Lz = params_in.getfloat('Lz', fallback=64) * wavelength_opt
else:
    Lx = params_in.getfloat('Lx', fallback=100)
    Ly = params_in.getfloat('Ly', fallback=50)
    Lz = params_in.getfloat('Lz', fallback=800)
perturbation_amplitude = params_in.getfloat('perturbation_amplitude', fallback=1e-4)
random_seed = params_in.getint('random_seed', fallback=42)
stop_sim_time = params_in.getfloat('stop_sim_time', fallback=np.inf)
stop_wall_time = params_in.getfloat('stop_wall_time', fallback=np.inf)
stop_iteration = params_in.getfloat('stop_iteration', fallback=np.inf)
cfl_safety = params_in.getfloat('cfl_safety', fallback=0.3)
buoyancy_dtmax = params_in.getboolean('buoyancy_dtmax', fallback=False)
bdt_frac = params_in.getfloat('bdt_frac', fallback=5)
ncc_cutoff = params_in.getfloat('ncc_cutoff', fallback=1e-6)
mesh1 = params_in.getint('mesh1', fallback=0)
mesh2 = params_in.getint('mesh2', fallback=0)

# Processor mesh -- works if ncpu is a power of 2.
ncpu = MPI.COMM_WORLD.size
mesh = None
if mesh1*mesh2 == 0:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2 ** np.floor(log2 / 2)), int(2 ** np.ceil(log2 / 2))]
else:
    mesh = [mesh1, mesh2]

logger.info("running on processor mesh={}".format(mesh))


# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3 / 2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3 / 2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)
logger.info("domain built")
# 3D Boussinesq hydrodynamics
variables = ['T', 'C', 'w', 'u', 'v', 'p']

problem = de.IVP(domain, variables=variables, ncc_cutoff=ncc_cutoff)

# Parameters and backgrounds
problem.parameters['tau'] = tau
problem.parameters['Pr'] = Pr
problem.parameters['Sc'] = Pr / tau
problem.parameters['R0'] = R0
problem.parameters['Ra'] = 1/(R0*tau)
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
lamb_opt, kx_opt = eval_finger(R0, Pr, tau, rescale)
problem.parameters['k_opt'] = kx_opt

# Vector calc substitutions
problem.substitutions['Lap(A)'] = 'dx(dx(A)) + dy(dy(A)) + dz(dz(A))'
problem.substitutions['UdotGrad(A)'] = '(u*dx(A) + v*dy(A) + w*dz(A))'
problem.substitutions['Div(Cx, Cy, Cz)'] = '(dx(Cx) + dy(Cy) + dz(Cz))'

problem.substitutions['Ox'] = 'dy(w) - dz(v)'  # vorticity
problem.substitutions['Oy'] = 'dz(u) - dx(w)'
problem.substitutions['Oz'] = 'dx(v) - dy(u)'
problem.substitutions['b'] = 'T - C'
# Output operations
problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
problem.substitutions['vol_avg(A)'] = 'integ(A)/Lx/Ly/Lz'
problem.substitutions['prime(A)'] = 'A - plane_avg(A)'

# Add Equations
zero_cond = "(nx == 0) and (ny == 0) and (nz == 0)"
else_cond = "(nx != 0) or  (ny != 0) or  (nz != 0)"
problem.add_equation("Div(u, v, w) = 0", condition=else_cond)
problem.add_equation("p = 0", condition=zero_cond)
if rescale:
    problem.add_equation("dt(u) + Sc*(dx(p) - Lap(u)) = v*Oz - w*Oy")
    problem.add_equation("dt(v) + Sc*(dy(p) - Lap(v)) = w*Ox - u*Oz")
    if implicit_IGWs:
        problem.add_equation("dt(w) + Sc*(dz(p) - Lap(w) - (T - C)) = u*Oy - v*Ox")
        problem.add_equation("dt(T) + (w - Lap(T))/tau = -UdotGrad(T)")
        problem.add_equation("dt(C) + Ra*w - Lap(C) = -UdotGrad(C)")
    else:
        problem.add_equation("dt(w) + Sc*(dz(p) - Lap(w)) = u*Oy - v*Ox + Sc*(T - C)")
        problem.add_equation("dt(T) - Lap(T)/tau = -UdotGrad(T) - w/tau")
        problem.add_equation("dt(C) - Lap(C) = -UdotGrad(C) - Ra*w")
else:
    problem.add_equation("dt(u)  + Pr*dx(p) - Pr*Lap(u) = v*Oz - w*Oy")
    problem.add_equation("dt(v)  + Pr*dy(p) - Pr*Lap(v) = w*Ox - u*Oz")
    if implicit_IGWs:
        problem.add_equation("dt(w)  + Pr*dz(p) - Pr*Lap(w) - Pr*(T - C) = u*Oy - v*Ox")
        problem.add_equation("dt(T) + w - Lap(T) = -UdotGrad(T)")
        problem.add_equation("dt(C) + w/R0 - tau*Lap(C) = -UdotGrad(C)")
    else:
        problem.add_equation("dt(w)  + Pr*dz(p) - Pr*Lap(w) = u*Oy - v*Ox + Pr*(T - C)")
        problem.add_equation("dt(T) - Lap(T) = -UdotGrad(T) - w")
        problem.add_equation("dt(C) - tau*Lap(C) = -UdotGrad(C) - w/R0")
x_basis, y_basis, z_basis = domain.bases

logger.info("problem built")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
# Initial conditions or restart
if not Path(basedir / 'restart.h5').exists():
    x, y, z = domain.all_grids()
    C = solver.state['C']

    # Random perturbations, initialized globally for same results in parallel
    ic_shape = np.array((16, 16, 16))  # only put noise in the first 16 coeffs in each direction.
    ic_scales = tuple(ic_shape / np.array((Nx, Ny, Nz)))
    gshape = domain.dist.grid_layout.global_shape(scales=ic_scales)
    slices = domain.dist.grid_layout.slices(scales=ic_scales)
    rand = np.random.RandomState(seed=random_seed)
    noise = domain.new_field()
    noise.set_scales(ic_scales)
    noise['g'] = rand.standard_normal(gshape)[slices]
    noise.set_scales(1)
    pert = perturbation_amplitude * noise['g']
    C['g'] = pert

    fh_mode = 'overwrite'
else:
    # Restart
    write, last_dt = solver.load_state(basedir / 'restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
checkpoints = solver.evaluator.add_file_handler(basedir / 'checkpoint', wall_dt=3500, max_writes=1, mode=fh_mode)
checkpoints.add_system(solver.state)

snapshots = solver.evaluator.add_file_handler(basedir / 'slices', sim_dt=0.025/lambda_opt, max_writes=200, mode=fh_mode)
for field in ['u', 'v', 'w', 'T', 'C', 'b']:
    # for xv, label in zip((0, params['Lx'] / 2), ('xwall', 'xmid')):
    #     snapshots.add_task("{}(x={})".format(field, xv), name='{}_{}'.format(field, label))
    snapshots.add_task("{}(x=0)".format(field), name='{}_xwall'.format(field))

    # for yv, label in zip((0, params['Ly'] / 2), ('ywall', 'ymid')):
    #     snapshots.add_task("{}(y={})".format(field, yv), name='{}_{}'.format(field, label))
    snapshots.add_task("{}(y=0)".format(field), name='{}_ywall'.format(field))

    for zv, label in zip((0, Lz / 2), ('zwall', 'zmid')):
        snapshots.add_task("{}(z={})".format(field, zv), name='{}_{}'.format(field, label))

# spectra = solver.evaluator.add_file_handler(basedir / 'spectra', sim_dt=0.025 / lambda_opt, max_writes=200, mode=fh_mode)
# for field in ['u', 'v', 'w', 'T', 'C', 'b']:
#     spectra.add_task('integ({}, "y")/Ly'.format(field), name='y-avg_{}'.format(field))
#     # spectra.add_task('integ({}*sin(k_opt*y), "y")/Ly'.format(field), name='y-sin_k_opt_y-avg_{}'.format(field))
#     # spectra.add_task('integ({}*cos(k_opt*y), "y")/Ly'.format(field), name='y-cos_k_opt_y-avg_{}'.format(field))
#     spectra.add_task('integ({}, "x")/Lx'.format(field), name='x-avg_{}'.format(field))
#     spectra.add_task('integ({}, "z")/Lz'.format(field), name='z-avg_{}'.format(field))

scalars = solver.evaluator.add_file_handler(basedir / 'scalars', sim_dt=min(2, 1 / lambda_opt), max_writes=int(1e6), mode=fh_mode)
scalars.add_task("sqrt(vol_avg(u**2 + v**2))", name='u_perp_rms')
scalars.add_task("sqrt(vol_avg(prime(u)**2 + prime(v)**2))", name='u_prime_perp_rms')
scalars.add_task("sqrt(vol_avg(w**2))", name='w_rms')
scalars.add_task("sqrt(vol_avg(T**2))", name='T_rms')
scalars.add_task("sqrt(vol_avg(C**2))", name='C_rms')
scalars.add_task("vol_avg(Ox**2 + Oy**2 + Oz**2)", name='enstrophy')
scalars.add_task("vol_avg(prime(Ox)**2 + prime(Oy)**2 + prime(Oz)**2)", name='enstrophy_prime')
scalars.add_task("vol_avg(w*T)", name='FT')
scalars.add_task("vol_avg(w*C)", name='FC')

profiles = solver.evaluator.add_file_handler(basedir / 'profiles', sim_dt=0.1/lambda_opt, max_writes=int(1e6), mode=fh_mode)

profiles.add_task("plane_avg(u)", name='u_bar')
profiles.add_task("plane_avg(v)", name='v_bar')
profiles.add_task("plane_avg(T)", name='T_bar')
profiles.add_task("plane_avg(C)", name='C_bar')

# full_cube = solver.evaluator.add_file_handler(basedir / 'full_cube', sim_dt=2.5/lambda_opt, max_writes=10, mode=fh_mode)
# full_cube.add_task("u", name='u')
# full_cube.add_task("v", name='v')
# full_cube.add_task("w", name='w')
# full_cube.add_task("T", name='T')
# full_cube.add_task("C", name='C')

# CFL
# the IGW frequency at large scales in these units is given by
# buoy_freq = sqrt(Sc*(-Ra + 1/tau))
# Since we're treating the relevant terms implicitly, stepping over that frequency is numerically stable, but means
# those waves will be artificially damped. But we're working in a limit where they're likely critically damped due to
# physical effects anyway. So it shouldn't be crucial to capture this frequency, but it's something worth testing.
# buoy_freq = np.sqrt(Sc*(-Ra + 1/tau))
if rescale:
    buoy_freq = np.sqrt(Sc*(-Ra + 1/tau))
else:
    buoy_freq = np.sqrt(Pr * (1.0 - 1.0/R0))
t_buoy = 2.0*np.pi / buoy_freq
if buoyancy_dtmax:
    max_dt = t_buoy / bdt_frac
else:
    max_dt = 1/lambda_opt  # probably way too small if Ra is very small
if fh_mode == 'overwrite':
    dt = 0.25*max_dt
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u**2 + v**2 + w**2)", name='Re')

# Main loop
# dt = CFL.compute_dt()
# dt = solver.step(dt)
try:
    finite_Re = True
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration - 1) % flow.cadence == 0:
            out_string = 'Iteration: %i, Time: %e, dt: %e' % (solver.iteration, solver.sim_time, dt)
            out_string += '; Max Re = %f' % flow.max('Re')
            logger.info(out_string)

            if not np.isfinite(flow.max('Re')):
                logger.info('breaking with non-finite Re')
                finite_Re = False
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size))

    final_checkpoint = solver.evaluator.add_file_handler(basedir / 'final_checkpoint', iter=1, mode=fh_mode,
                                                         max_writes=1)
    final_checkpoint.add_system(solver.state)
    solver.step(dt)  # clean this up in the future...works for now.
    # post.merge_process_files(out_dir + '/final_checkpoint/', cleanup=False)

    # for task in file_handlers:
    #     post.merge_analysis(task.base_path)
