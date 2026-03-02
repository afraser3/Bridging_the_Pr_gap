import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import glob
from configparser import ConfigParser
from pathlib import Path

plt.rcParams.update({"text.usetex": True})


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


def r_from_R(R0, tau):
    return (R0 - 1)/(-1 + 1/tau)


def R0_from_r(r, tau):
    return 1 + r*(-1 + 1/tau)


def calc_Re(R0, tau, Pr, BGS13_est=False):
    if R0 >= 1/tau:
        return 0
    else:
        if BGS13_est:
            r = r_from_R(R0, tau)
            if r <= tau:
                return np.sqrt((1 + tau/Pr)/Pr)
            elif r > tau and r < 0.3:
                return np.sqrt((tau/Pr)*(1 + tau/Pr)/r)
            else:
                return (tau/Pr)*(1 - r)
        else:
            lam_f, l_f = eval_finger(R0, Pr, tau)
            return lam_f/(Pr*l_f**2)


def parse_cfg_file(run_name, subdir = ''):
    cfg_file = 'config_files/' + subdir + run_name + '.cfg'
    config_file = Path(cfg_file)
    runconfig = ConfigParser()
    runconfig.read(str(config_file))
    params_in = runconfig['parameters']
    params = dict()
    kopt_Lscale = params_in.getboolean('kopt_Lscale', fallback=False)
    tau = params['tau'] = params_in.getfloat('tau', fallback=0.05)
    Pr = params['Pr'] = params_in.getfloat('Pr', fallback=5)
    Sc = params['Pr'] / params['tau']
    rescale = params['rescale'] = params_in.getboolean('rescale', fallback=False)
    R0_is_epsilon = params_in.getboolean('R0_is_epsilon',
                                         fallback=False)  # the parameter named "R0" in the input file is actually supercriticality ("epsilon")
    reduced_r = params_in.getboolean('reduced_r',
                                     fallback=False)  # the parameter named "R0" in the input file is actually r ("reduced density ratio")
    if R0_is_epsilon and reduced_r:
        raise ValueError("R0_is_epsilon and reduced_r cannot both be True")
    elif R0_is_epsilon:
        epsilon = params_in.getfloat('R0', fallback=0.1)
        Rayleigh = epsilon + 1  # see eqn 2.1 of Fraser et al. 2025 (helical flows paper)
        R0 = params['R0'] = 1 / (Rayleigh * tau)
    elif reduced_r:
        R0 = params['R0'] = R0_from_r(params_in.getfloat('R0', fallback=2), tau)
    else:
        R0 = params['R0'] = params_in.getfloat('R0', fallback=2)
    Ra = 1 / (R0 * tau)
    lambda_opt, k_opt = eval_finger(R0, params['Pr'], params['tau'], rescale)
    wavelength_opt = 2 * np.pi / k_opt
    params['k_opt'] = k_opt
    params['wavelength_opt'] = wavelength_opt
    params['lambda_opt'] = lambda_opt
    params['Nx'] = params_in.getint('Nx', fallback=64)
    params['Ny'] = params_in.getint('Ny', fallback=64)
    params['Nz'] = params_in.getint('Nz', fallback=64)
    if kopt_Lscale:
        params['Lx'] = params_in.getfloat('Lx', fallback=8) * wavelength_opt
        params['Ly'] = params_in.getfloat('Ly', fallback=4) * wavelength_opt
        params['Lz'] = params_in.getfloat('Lz', fallback=64) * wavelength_opt
    else:
        params['Lx'] = params_in.getfloat('Lx', fallback=100)
        params['Ly'] = params_in.getfloat('Ly', fallback=50)
        params['Lz'] = params_in.getfloat('Lz', fallback=800)
    # return params
    return params['R0'], params['tau']


taumin = 1e-7

# R0s = np.geomspace(1 + 0.1*taumin, 1/taumin, endpoint=False)
R0s = np.append(np.geomspace(1 + 0.1*taumin, 2-taumin, endpoint=True, num=50), np.geomspace(2, 1/taumin, endpoint=False, num=25))
Les = np.geomspace(1, 1/taumin)
taus = 1/Les
Prs = 2*taus
# r_scan = np.geomspace(taumin, 1, endpoint=False)
# r_scan = np.geomspace(r_from_R(R0s[0], taus[-1]), 1, endpoint=False)

Res = np.zeros((len(R0s), len(taus)), dtype=np.float64)
Pes = np.zeros((len(R0s), len(taus)), dtype=np.float64)
# rs = np.zeros_like(Res)

for ri, r0 in enumerate(R0s):
    for ti, tau in enumerate(taus):
        if r0 < 1/tau:
            # rs[ri, ti] = r_from_R(r0, tau)
            Res[ri, ti] = calc_Re(r0, tau, Prs[ti])
            Pes[ri, ti] = Res[ri, ti] * tau


scale = 0.6
fig = plt.figure(figsize=(6*scale, 5*scale))
ax1 = plt.gca()

ax1.set_xscale('log')
ax1.set_yscale('log')
pcm1 = ax1.contourf(R0s, taus, 1e2*Res.T, cmap='hot_r', norm=colors.LogNorm(vmin=1e1), levels=np.geomspace(1e1, 1e6, 6))#, alpha=0)
# pcm1 = ax1.contourf(R0s, taus, 1e2*Pes.T, cmap='hot_r', norm=colors.LogNorm(vmin=1e-7, vmax=1), levels=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])#, alpha=0)
# ax1.yaxis.set_inverted(True)
# ax1.set_xlabel(r'Stratification ($R_0 \propto dT_0/dz$)')
ax1.set_xlabel(r'Density ratio $R_0 \equiv |N_T^2|/|N_C^2|$')
ax1.set_ylabel(r'Diffusivity ratio $\tau \equiv \kappa_C/\kappa_T$')
# ax1.set_ylabel(r'Thermal diffusivity ($\kappa_T/\kappa_S$)')
# cbar1 = fig.colorbar(pcm1, ax=ax1, label='$\mathrm{Re}(R_0, \tau)$')
cbar1 = fig.colorbar(pcm1, ax=ax1, label='Estimated Reynolds number')
# ax1.contour(R0s, taus, 1e2*Pes.T, levels=[1], colors='cyan')

run_names1 = [Path(fpath).stem for fpath in glob.glob('runs/*tau0p05*')]
run_names2 = [Path(fpath).stem for fpath in glob.glob('runs/*tau5e-4*')]
run_names3 = [Path(fpath).stem for fpath in glob.glob('runs/*tau5e-7*')]
# run_names1.remove('r0p1_tau0p05_Pr0p1_nobdt')
run_names3.remove('eps100_tau5e-7_Pr1e-6_hres')
# run_names3.remove('eps1e3_tau5e-7_Pr1e-6_vhres')
# run_names3.remove('eps1e3_tau5e-7_Pr1e-6_vvhres')
run_names3.remove('eps300_tau5e-7_Pr1e-6_vvhres')
run_names3.remove('eps300_tau5e-7_Pr1e-6_vvhresx1p5')
#
run_names3.remove('eps1_tau5e-7_Pr1e-6_no-rs_lin')
run_names3.remove('eps1_tau5e-7_Pr1e-6_with-rs')
run_names3.remove('eps1_tau5e-7_Pr1e-6_with-rs_lin')
run_names3.remove('eps1_tau5e-7_Pr1e-6_no-rs')
run_names3.remove('eps1_tau5e-7_Pr1e-6_no-rs2')
run_names3.remove('eps0p1_tau5e-7_Pr1e-6_no-rs2')
run_names3.remove('eps0p1_tau5e-7_Pr1e-6_no-rs3')
run_names3.remove('eps0p1_tau5e-7_Pr1e-6')
run_names3.remove('eps0p1_tau5e-7_Pr1e-6_2')
run_names3.remove('eps10_tau5e-7_Pr1e-6_hres')
run_names4 = [Path(fpath).stem for fpath in glob.glob('runs/*tau5e-3*')]

R0s1, taus1 = np.array([parse_cfg_file(run_name) for run_name in run_names1]).T
R0s2, taus2 = np.array([parse_cfg_file(run_name) for run_name in run_names2]).T
R0s3, taus3 = np.array([parse_cfg_file(run_name) for run_name in run_names3]).T
R0s4, taus4 = np.array([parse_cfg_file(run_name) for run_name in run_names4]).T
# data1 = [parse_cfg_file(run_name) for run_name in run_names1]
# data2 = [parse_cfg_file(run_name) for run_name in run_names2]
# data3 = [parse_cfg_file(run_name) for run_name in run_names3]
# data4 = [parse_cfg_file(run_name) for run_name in run_names4]
# print(taus4)
# print(R0s4)

# plot DNS data
# ax1.plot(data[:, 0], data[:, 3], '.', c='C0', ms=10)
ax1.plot(R0s1, taus1, '.', c='k', ms=10)
ax1.plot(R0s2, taus2, '.', c='k', ms=10)
ax1.plot(R0s3, taus3, '.', c='k', ms=10)
ax1.plot(R0s4, taus4, '.', c='k', ms=10)

# ax1.set_ylim(ymin=taumin)
# ax1.plot(R0s, 1/R0s, c='k')

#  "No instability" region label and hatches
# ax1.fill_between(R0s, 1/R0s, 1, hatch='/', alpha=0)
# ax1.text(2e3, 2e-4, 'No instability', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, rotation=-45, fontsize='large')

# ax1.fill_between([0.7e1, 1e6], [0.3e-3, 0.5e-6], [0.5e-6, 0.5e-6], color='green', alpha=0.5)
R01 = 3
tau_1 = 1e-3
tau_2 = 2e-7
R02 = 1/(5*tau_1)
R03 = 1/(5*tau_2)
# ax1.fill_between([R01, R02, R03], [tau_1, tau_1, tau_2], [tau_2, tau_2, tau_2], color='green', alpha=0.5)
# ax1.text(5e1, 1e-5, 'Stars', color='white', weight='bold')
# ax1.text(1, 4e-3, 'Oceans,\nPlanets', color='white', bbox={'facecolor': 'blue', 'alpha': 0.75, 'pad': 2}, fontsize=10, weight='bold')
# 1/tau version
ax1.fill_between([R01, R02, R03], [tau_1, tau_1, tau_2], [tau_2, tau_2, tau_2], color='green', alpha=0.5)
ax1.text(2e1, 1/1e5, r'\textbf{Stars}', color='white', weight='bold', fontsize=14)
# ax1.text(5e1, 1/1e5, 'Stars', color='white', weight='bold')
# ax1.text(1, 1/0.25e3, r'\noindent Oceans,\newline \noindent Planets', color='white', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 2}, fontsize=10, weight='bold')
# ax1.text(1, 1/0.25e3, 'Oceans,\n Planets', color='white', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 2}, fontsize=10, weight='bold')
ax1.text(1.5, 1e-2, r'\textbf{Oceans}', color='white', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 2}, fontsize=12, weight='bold')

# ax1.fill_between([R01, R02, R03], [tau_1, tau_1, tau_2], [tau_2, tau_2, tau_2], color='green', alpha=0.25)
# ax1.text(1, 4e-3, 'Oceans,\nPlanets', color='white', bbox={'facecolor': 'blue', 'alpha': 0.25, 'pad': 2}, fontsize=10, weight='bold', alpha=0)

# ax1.fill_between([R01, R02, R03], [1/tau_1, 1/tau_1, 1/tau_2], [1/tau_2, 1/tau_2, 1/tau_2], color='green', alpha=0.25)
# ax1.text(1, 0.25e3, 'Oceans,\nPlanets', color='white', bbox={'facecolor': 'blue', 'alpha': 0.25, 'pad': 2}, fontsize=10, weight='bold', alpha=0)
# ax1.set_title(r'Regimes of thermohaline convection')
plt.tight_layout()
# plt.show()
plt.savefig('figures/Re_tau_R0_diagram.pdf', bbox_inches='tight')

# old, irrelevant
# ax1.text(1, 1e-2, 'Oceans', color='white', bbox={'facecolor': 'blue', 'alpha': 0.75, 'pad': 2}, fontsize=10, weight='bold')
# ax1.text(1e1, 1e-4, ' WDs ', color='white', bbox={'facecolor': 'grey', 'alpha': 0.65, 'pad': 0.75, 'boxstyle': 'ellipse'}, fontsize=10, weight='bold')
# ax1.text(3e2, 1e-6, '  RGBs  ', color='white', bbox={'facecolor': 'red', 'alpha': 0.65, 'pad': 0.75, 'boxstyle': 'ellipse'}, fontsize=10, weight='bold')

# ax1.text(1, 1e-2, 'Oceans', bbox={'facecolor': 'blue', 'alpha': 0.25, 'pad': 2}, fontsize=10, weight='bold', alpha=0)
# ax1.text(1e1, 1e-4, ' WDs ', bbox={'facecolor': 'grey', 'alpha': 0.25, 'pad': 0.75, 'boxstyle': 'ellipse'}, fontsize=10, weight='bold', alpha=0)
# ax1.text(3e2, 1e-6, '  RGBs  ', bbox={'facecolor': 'red', 'alpha': 0.25, 'pad': 0.75, 'boxstyle': 'ellipse'}, fontsize=10, weight='bold', alpha=0)