import numpy as np
import h5py
import glob
from configparser import ConfigParser
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


def eval_finger(R0, Pr, tau, rescale=False):
    # Evaluate the growth rate and horizontal wavenumber of fastest-growing
    # elevator mode. This uses the fingering-mode notes developed by Rich Townsend

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


def eval_finger_LPN(Ra, Sc):
    # Evaluate the growth rate and horizontal wavenumber of fastest-growing
    # elevator mode. This uses the fingering-mode notes developed by Rich Townsend

    a3 = 1
    a2 = 2*Sc + 2
    a1 = -(Ra - 4*Sc - 1 + Sc*Ra)
    a0 = 2*Sc*(1 - Ra)

    p = a1 * a2 / (6 * a3 ** 2) - a0 / (2 * a3) - (a2 / (3 * a3)) ** 3
    q = (a2 / (3 * a3)) ** 2 - a1 / (3 * a3)

    tlam = 2 * np.sqrt(q) * np.cos(np.arccos(p / np.sqrt(q) ** 3) / 3) - a2 / (3 * a3)

    # l_f = (Pr * (1 + tlam - R0 * (tau + tlam)) / (R0 * (1 + tlam) * (Pr + tlam) * (tau + tlam))) ** 0.25
    l_f = ((Ra - 1 - tlam)*Sc/(tlam**2 + tlam + Sc*tlam + Sc))**0.25

    lam_f = l_f ** 2 * tlam

    return lam_f, l_f


def r_from_R(R, tau):
    return (R-1)/(-1 + 1/tau)


def R_from_r(r, tau):
    return r*(-1 + 1/tau) + 1


def epsilon_from_r(r, tau):
    R0 = R_from_r(r, tau)
    Rayleigh = 1/(R0*tau)
    return Rayleigh-1


def r_from_epsilon(vareps, tau):
    Rayleigh = vareps + 1
    R0 = 1/(Rayleigh*tau)
    return r_from_R(R0, tau)


def parse_cfg_file(cfg_file):
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
    Pm = params['Pm'] = params_in.getfloat('Pm', fallback=1)
    HB = params['HB'] = params_in.getfloat('HB', fallback=1)
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
        R0 = params['R0'] = R_from_r(params_in.getfloat('R0', fallback=2), tau)
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
    return params


def parse_LPN_cfg_file(cfg_file):
    config_file = Path(cfg_file)
    runconfig = ConfigParser()
    runconfig.read(str(config_file))
    params_in = runconfig['parameters']
    params = dict()
    kopt_Lscale = params_in.getboolean('kopt_Lscale', fallback=False)
    eps = params['eps'] = params_in.getfloat('eps', fallback=1)
    Ra = params['Ra'] = eps + 1
    Sc = params['Sc'] = params_in.getfloat('Sc', fallback=5)
    lambda_opt, k_opt = eval_finger_LPN(Ra, Sc)
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
    return params


def get_scalars(fname, task_names):
    with h5py.File(fname, 'r') as f:
        sim_time = np.array(f['scales/sim_time'])
        tasks = [np.squeeze(np.array(f['tasks/{}'.format(task_name)])) for task_name in task_names]
    return sim_time, tasks


def glob_scalars(run_dir, task_names):
    # check how many scalars_s*.h5 files there are in this directory and stitch them together
    last_ind = max([int(fpath.split('scalars_s')[-1][0]) for fpath in glob.glob(run_dir + '*.h5')])
    names = [run_dir + 'scalars_s{}.h5'.format(i) for i in range(1, last_ind + 1)]
    for n, name in enumerate(names):
        if n==0:
            sim_time, tasks = get_scalars(name, task_names)
        else:
            sim_time2, tasks2 = get_scalars(name, task_names)
            if sim_time2[0] < sim_time[-1]:  # check for redundant entries
                extra_ind = np.argmin(np.abs(sim_time - sim_time2[0]))  # index in sim_time that coincides with first redundant datapoint
                if extra_ind == 0:
                    print("reran checkpoint from t=0, so redundant files exist")
                    sim_time = sim_time2
                    tasks = tasks2
                else:
                    sim_time = np.append(sim_time[:extra_ind], sim_time2)
                    # Srms = np.append(Srms[:extra_ind], Srms2)
                    for i in range(len(task_names)):
                        tasks[i] = np.append(tasks[i][:extra_ind], tasks2[i])
            else:
                sim_time = np.append(sim_time, sim_time2)
                # Srms = np.append(Srms, Srms2)
                for i in range(len(task_names)):
                    tasks[i] = np.append(tasks[i], tasks2[i])
    return sim_time, tasks


# run_names = [Path(fpath).stem for fpath in glob.glob('config_files/*tau*')]
# run_names.remove('eps10_tau5e-7_Pr1e-6_hres')

# run_names = [Path(fpath).stem for fpath in glob.glob('runs/*')]
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
run_names_LPN = [Path(fpath).stem for fpath in glob.glob('runs/hydro_LPN/eps*')]

plot_both_panels = True


def get_data_from_run(run_name):
    Run_dir = 'runs/' + run_name + '/'
    cfg_file = 'config_files/' + run_name + '.cfg'
    params = parse_cfg_file(cfg_file)
    print(Run_dir)
    times, scalars = glob_scalars(Run_dir + 'scalars/', ['w_rms', 'FC', 'FT'])
    wrms = np.mean(scalars[0][int(3*len(times) / 4):])
    FC = np.mean(scalars[1][int(3*len(times) / 4):])
    if (FC - scalars[1][-1])/scalars[1][-1] > 0.05:
        raise ValueError("check whether "+run_name+" has been run long enough")
    R0 = params['R0']
    tau = params['tau']
    Pr = params['Pr']
    r = r_from_R(R0, tau)
    eps = epsilon_from_r(r, tau)
    lambda_opt = params['lambda_opt']
    lambda_opt_PG = lambda_opt*tau
    k_opt = params['k_opt']
    w_BGS13 = 2 * np.pi * lambda_opt / k_opt
    FC_BGS13 = 49 * lambda_opt_PG**2 / (k_opt**2 * R0 * (lambda_opt_PG + tau * k_opt**2)) / tau**2
    return R0, r, eps, tau, Pr, wrms, w_BGS13, np.abs(FC), FC_BGS13


def get_data_from_LPN_run(run_name):
    Run_dir = 'runs/hydro_LPN/' + run_name + '/'
    cfg_file = 'config_files/hydro_LPN/' + run_name + '.cfg'
    params = parse_LPN_cfg_file(cfg_file)
    times, scalars = glob_scalars(Run_dir + 'scalars/', ['w_rms', 'FC'])
    wrms = np.mean(scalars[0][int(len(times) / 2):])
    FC = np.mean(scalars[1][int(len(times) / 2):])
    eps = params['eps']
    Sc = params['Sc']
    lambda_opt = params['lambda_opt']
    # lambda_opt_PG = lambda_opt*tau
    k_opt = params['k_opt']
    w_BGS13 = 2 * np.pi * lambda_opt / k_opt
    # FC_BGS13 = 49 * lambda_opt_PG**2 / (k_opt**2 * R0 * (lambda_opt_PG + tau * k_opt**2)) / tau**2
    return eps, Sc, wrms, w_BGS13, np.abs(FC)


r_scan = np.append(np.geomspace(1e-4, 0.1, num=100, endpoint=False), np.linspace(0.1, 1, endpoint=False))
# epsilon_scan = np.geomspace(epsilon_from_r(np.max(r_scan), 5e-7), 1e5, num=500)
epsilon_scan = np.geomspace(1e-4, 1e5, num=500)


def w_BGS13_scan(rs, Pr, tau):
    wscan = np.zeros_like(rs)
    for ri, r in enumerate(rs):
        R0 = R_from_r(r, tau)
        lopt, kopt = eval_finger(R0, Pr, tau, True)
        wscan[ri] = 2*np.pi*lopt/kopt
    return wscan


def w_BGS13_LPN_scan(epsilons, Sc):
    wscan = np.zeros_like(epsilons)
    for ei, eps in enumerate(epsilons):
        lopt, kopt = eval_finger_LPN(eps + 1, Sc)
        wscan[ei] = 2*np.pi*lopt/kopt
    return wscan


def FC_BGS13_scan(rs, Pr, tau):
    FC_scan = np.zeros_like(rs)
    for ri, r in enumerate(rs):
        R0 = R_from_r(r, tau)
        eps = epsilon_from_r(r, tau)
        lopt, kopt = eval_finger(R0, Pr, tau, True)
        # FC_scan[ri] = - 49 * lopt**2 / (R0 * kopt**2 * (lopt + tau * kopt**2) * tau**2)
        FC_scan[ri] = - 49 * (eps + 1) * lopt ** 2 / (kopt ** 2 * (lopt + kopt ** 2))
    return -FC_scan


def FC_BGS13_LPN_scan(epsilons, Sc):
    FC_scan = np.zeros_like(epsilons)
    for ei, eps in enumerate(epsilons):
        lopt, kopt = eval_finger_LPN(eps + 1, Sc)
        # FC_scan[ri] = - 49 * lopt**2 / (R0 * kopt**2 * (lopt + tau * kopt**2) * tau**2)
        FC_scan[ei] = -49 * (eps + 1) * lopt**2 / (kopt ** 2 * (lopt + kopt**2))
    return -FC_scan


def FT_BGS13_scan(rs, Pr, tau):
    FT_scan = np.zeros_like(rs)
    for ri, r in enumerate(rs):
        R0 = R_from_r(r, tau)
        lopt, kopt = eval_finger(R0, Pr, tau, True)
        FT_scan[ri] = - 49 * lopt**2 / (kopt**2 * (lopt + kopt**2) * tau**2)
    return FT_scan


# data = np.array([get_data_from_run(rn) for rn in run_names])
data1 = np.array([get_data_from_run(rn) for rn in run_names1])
data2 = np.array([get_data_from_run(rn) for rn in run_names2])
data3 = np.array([get_data_from_run(rn) for rn in run_names3])
data4 = np.array([get_data_from_run(rn) for rn in run_names4])
data_LPN = np.array([get_data_from_LPN_run(rn) for rn in run_names_LPN])
sortind1 = np.argsort(data1[:,0])
data1 = data1[sortind1, :]
sortind2 = np.argsort(data2[:,0])
data2 = data2[sortind2, :]
sortind3 = np.argsort(data3[:,0])
data3 = data3[sortind3, :]
sortind4 = np.argsort(data4[:,0])
data4 = data4[sortind4, :]
sortind_LPN = np.argsort(data_LPN[:,0])
data_LPN = data_LPN[sortind_LPN, :]

xaxis_ind = 1  # for r
# yaxis_ind = 5  # for wrms
# yaxis_LPN_ind = 2  # for wrms
yaxis_ind = 7  # for FC
yaxis_LPN_ind = 4  # for FC
scale = 0.6
if plot_both_panels:
    data_labels = [r'$\mathrm{Pr} = 10^{-1}$', r'$\mathrm{Pr} = 10^{-2}$',
                   r'$\mathrm{Pr} = 10^{-3}$', r'$\mathrm{Pr} = 10^{-6}$']
    # BGS13_labels = [r'$\mathrm{Pr} = 10^{-1}$ BGS13', r'$\mathrm{Pr} = 10^{-2}$ BGS13',
    #                 r'$\mathrm{Pr} = 10^{-3}$ BGS13', r'$\mathrm{Pr} = 10^{-6}$ BGS13']
    BGS13_labels = ['']*4
    plt.figure(figsize=(14*scale, 5*scale))
    plt.subplot(1, 2, 1)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.plot(data1[:, xaxis_ind], data1[:, yaxis_ind], '.', c='C0', label=data_labels[0])
    plt.plot(data4[:, xaxis_ind], data4[:, yaxis_ind], '^', c='C4', label=data_labels[1])
    plt.plot(data2[:, xaxis_ind], data2[:, yaxis_ind], 'x', c='C1', label=data_labels[2])
    plt.plot(data3[:, xaxis_ind], data3[:, yaxis_ind], '+', c='C2', label=data_labels[3])
    plt.plot(1 / (data_LPN[:, 0] + 1), data_LPN[:, yaxis_LPN_ind], 'o', c='k', label=r'$\mathrm{Pr} = 0$', zorder=0)

    plt.plot(r_scan, FC_BGS13_scan(r_scan, 1e-1, 5e-2), c='C0', label=BGS13_labels[0])
    plt.plot(r_scan, FC_BGS13_scan(r_scan, 1e-2, 5e-3), c='C4', label=BGS13_labels[1])
    plt.plot(r_scan, FC_BGS13_scan(r_scan, 1e-3, 5e-4), '--', c='C1', label=BGS13_labels[2])
    plt.plot(r_scan, FC_BGS13_scan(r_scan, 1e-6, 5e-7), ':', c='C2', label=BGS13_labels[3])
    # plt.plot(data1[:, xaxis_ind], data1[:, yaxis_ind + 1], c='C0', label=r'$\mathrm{Pr} = 10^{-1}$ BGS13')
    # plt.plot(data4[:, xaxis_ind], data4[:, yaxis_ind + 1], c='C4', label=r'$\mathrm{Pr} = 10^{-2}$ BGS13')
    # plt.plot(data2[:, xaxis_ind], data2[:, yaxis_ind + 1], c='C1', label=r'$\mathrm{Pr} = 10^{-3}$ BGS13')
    # plt.plot(data3[:, xaxis_ind], data3[:, yaxis_ind + 1], '--', c='C2', label=r'$\mathrm{Pr} = 10^{-6}$ BGS13')
    # plt.plot(data3[-4:, xaxis_ind], 5*data3[-4:, xaxis_ind]**(5/4), '--', c='k')

    # plt.plot(1 / (epsilon_scan + 1), w_BGS13_LPN_scan(epsilon_scan, 2), '-', c='k', linewidth=3, label=r'$\mathrm{Pr} \to 0$ BGS13', zorder=0)
    plt.plot(1 / (epsilon_scan + 1), FC_BGS13_LPN_scan(epsilon_scan, 2), '-', c='k', linewidth=3, zorder=0)
    # plt.legend(ncol=2)
    first_legend = plt.legend(ncol=2, loc='lower right')
    plt.gca().add_artist(first_legend)

    DNS_dummy = mlines.Line2D([], [], linestyle='None', marker='.', c='grey', label=r'DNS')
    BGS13_dummy = mlines.Line2D([], [], linestyle='-', c='grey', label=r'BGS13 model')
    second_legend = plt.legend(handles=[DNS_dummy, BGS13_dummy], loc='upper left')

    # plt.ylabel(r'$\hat{w}_\mathrm{rms}/\tau$')
    plt.ylabel(r'$\tilde{F}_C$')
    # plt.ylabel(r'$\hat{D}_C/(R_0 \tau^2)$')
    plt.xlabel(r'reduced density ratio $r$')
    # plt.xlabel(r'$(R_0 \tau)^{-1} - 1$')
    # plt.xlim((0, 1))
    # plt.xlim(xmax=1)
    # plt.xlim(xmin=1e-3)
    plt.xlim((1, 1e-3))
    # plt.ylim((1e-1, 3e2))
    # plt.ylim((2e-1, 3e2))
    plt.ylim((1e-1, 3e6))

    plt.subplot(1, 2, 2)
    data_labels = ['', '', '']
    BGS13_labels = ['', '', '']
else:
    plt.figure(figsize=(7 * scale, 5 * scale))
    # plt.ylabel(r'$\hat{w}_\mathrm{rms}/\tau$')
    # plt.ylabel(r'$w_\mathrm{rms}$')
    plt.ylabel(r'$\tilde{F}_C$')
    # data_labels = [r'$\mathrm{Pr} = 10^{-1}$ DNS', r'$\mathrm{Pr} = 10^{-2}$ DNS', r'$\mathrm{Pr} = 10^{-3}$ DNS', r'$\mathrm{Pr} = 10^{-6}$ DNS']
    # BGS13_labels = [r'$\mathrm{Pr} = 10^{-1}$ BGS13', r'$\mathrm{Pr} = 10^{-2}$ BGS13', r'$\mathrm{Pr} = 10^{-3}$ BGS13', r'$\mathrm{Pr} = 10^{-6}$ BGS13']
    # data_labels = [r'$\mathrm{Pr} = 10^{-1}$', r'$\mathrm{Pr} = 10^{-3}$', r'$\mathrm{Pr} = 10^{-6}$']
    data_labels = [r'$\tau = 5 \times 10^{-2}$', r'$\tau = 5 \times 10^{-4}$', r'$\tau = 5 \times 10^{-7}$']
    BGS13_labels = ['', '', '']


xaxis_ind = 2
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')

rescale_exp = 0

plt.plot(epsilon_scan, FC_BGS13_LPN_scan(epsilon_scan, 2)/epsilon_scan**rescale_exp, '-', c='k', linewidth=3)#, label=r'$\mathrm{Pr} \to 0$ BGS13')
plt.plot(data_LPN[:, 0], data_LPN[:, yaxis_LPN_ind]/data_LPN[:, 0]**rescale_exp, 'o', c='k', label=r'$\tau \to 0$')
# plt.plot(data_LPN[:, 0], 15*data_LPN[:, 0]**1.25, '--', c='k')
# plt.plot(data_LPN[:, 0], 2*data_LPN[:, 0], '--', c='red')

plt.plot(data1[:, xaxis_ind], data1[:, yaxis_ind]/data1[:, xaxis_ind]**rescale_exp, '.', c='C0', label=data_labels[0])#, label=r'$\mathrm{Pr} = 10^{-1}$ DNS')
plt.plot(data4[:, xaxis_ind], data4[:, yaxis_ind]/data4[:, xaxis_ind]**rescale_exp, '^', c='C4')#, label=r'$\mathrm{Pr} = 10^{-2}$ DNS')
plt.plot(data2[:, xaxis_ind], data2[:, yaxis_ind]/data2[:, xaxis_ind]**rescale_exp, 'x', c='C1', label=data_labels[1])#, label=r'$\mathrm{Pr} = 10^{-3}$ DNS')
plt.plot(data3[:, xaxis_ind], data3[:, yaxis_ind]/data3[:, xaxis_ind]**rescale_exp, '+', c='C2', label=data_labels[2])#, label=r'$\mathrm{Pr} = 10^{-6}$ DNS')

# plt.plot(data_LPN[:, 0], data_LPN[:, 3]/data_LPN[:, 0]**rescale_exp, '-', c='k', label=r'$\mathrm{Pr} \to 0$ BGS13')

plt.plot(epsilon_from_r(r_scan, 5e-2), FC_BGS13_scan(r_scan, 1e-1, 5e-2)/epsilon_from_r(r_scan, 5e-2)**rescale_exp, c='C0', label=BGS13_labels[0])#, label=r'$\mathrm{Pr} = 10^{-1}$ BGS13')
plt.plot(epsilon_from_r(r_scan, 5e-3), FC_BGS13_scan(r_scan, 1e-2, 5e-3)/epsilon_from_r(r_scan, 5e-3)**rescale_exp, c='C4')#, label=r'$\mathrm{Pr} = 10^{-2}$ BGS13')
plt.plot(epsilon_from_r(r_scan, 5e-4), FC_BGS13_scan(r_scan, 1e-3, 5e-4)/epsilon_from_r(r_scan, 5e-4)**rescale_exp, '--', c='C1', label=BGS13_labels[1])#, label=r'$\mathrm{Pr} = 10^{-3}$ BGS13')
plt.plot(epsilon_from_r(r_scan, 5e-7), FC_BGS13_scan(r_scan, 1e-6, 5e-7)/epsilon_from_r(r_scan, 5e-7)**rescale_exp, ':', c='C2', label=BGS13_labels[2])#, label=r'$\mathrm{Pr} = 10^{-6}$ BGS13')
# plt.plot(data1[:, xaxis_ind], data1[:, yaxis_ind + 1]/data1[:, xaxis_ind]**rescale_exp, c='C0', label=r'$\mathrm{Pr} = 10^{-1}$ BGS13')
# plt.plot(data4[:, xaxis_ind], data4[:, yaxis_ind + 1]/data4[:, xaxis_ind]**rescale_exp, c='C4', label=r'$\mathrm{Pr} = 10^{-2}$ BGS13')
# plt.plot(data2[:, xaxis_ind], data2[:, yaxis_ind + 1]/data2[:, xaxis_ind]**rescale_exp, c='C1', label=r'$\mathrm{Pr} = 10^{-3}$ BGS13')
# plt.plot(data3[:, xaxis_ind], data3[:, yaxis_ind + 1]/data3[:, xaxis_ind]**rescale_exp, '--', c='C2', label=r'$\mathrm{Pr} = 10^{-6}$ BGS13')

# plt.xlim((1e-1, 3e6))
# plt.ylim(ymin=1e-1)
plt.xlim((1e-1, 1e3))
# plt.ylim((1e-1, 3e2))
# plt.ylim((2e-1, 3e2))
plt.ylim((1e-1, 3e6))
# first_legend = plt.legend(ncol=1, loc='lower right')
# plt.gca().add_artist(first_legend)

# Full_pts = mlines.Line2D([], [], marker='.', markersize=7.5, c='C0', label=r'Full model')
# IFSC_pts = mlines.Line2D([], [], marker='d', markersize=7.5, c='C2', label=r'IFSC model')
# IpFSC_pts = mlines.Line2D([], [], marker='x', markersize=7.5, c='C1', label=r'Modified IFSC model')
# handles, labels = axs[0,0].get_legend_handles_labels()
# first_legend = axs[0,0].legend(handles = [fit1, fit2, fit3, fit4], loc = 'upper left')
# first_legend = axs[0,0].legend(handles = [fit3, fit4], loc = 'lower right')
# axs[0,0].add_artist(first_legend)
# axs[0,0].legend(handles = [Full_pts, IFSC_pts, IpFSC_pts], loc = 'upper left')


# plt.axvline(epsilon_from_r(0, 0.05), c='C0')
# plt.axvline(epsilon_from_r(0, 5e-3), c='C4')
# plt.axvline(epsilon_from_r(0, 5e-4), c='C1')
# plt.axvline(epsilon_from_r(0, 5e-7), c='C2')
# plt.gca().invert_xaxis()
# plt.plot(data3[-4:, xaxis_ind], 5*data3[-4:, xaxis_ind]**(5/4), '--', c='k')
# plt.legend(ncol=2)
# plt.ylabel(r'$\hat{w}_\mathrm{rms}/\tau$')
# plt.xlabel(r'reduced density ratio $r$')
# plt.xlabel(r'supercriticality $(R_0 \tau)^{-1} - 1$')
plt.xlabel(r'supercriticality $\varepsilon = \mathcal{R} - 1 = (R_0 \tau)^{-1} - 1$', fontsize='large')
# plt.gca().set_yticks([])
plt.gca().yaxis.set_tick_params(which='both', labelleft=False)

plt.tight_layout()
# plt.savefig('BGS13_verification_wrms_TAMU.pdf', bbox_inches='tight')
plt.show()
