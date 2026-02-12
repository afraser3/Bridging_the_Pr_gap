"""
Compares a single scalar output between a set of runs. Specify which scalar (and whether to rescale it by some
prefactor) with "task" defined at the top. Runs to be compared are stored in "dirs".
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob


def eval_finger(R0, Pr, tau):
    # Evaluate the growth rate and horizontal wavenumber of fastest-growing
    # elevator mode of the full model. This uses the fingering-mode notes developed by Rich Townsend

    a3 = Pr * (1 - R0) + tau - R0
    a2 = -2 * (R0 - 1) * (Pr + tau + Pr * tau)
    a1 = Pr + tau - 4 * Pr * (R0 - 1) * tau - (1 + Pr) * R0 * tau ** 2
    a0 = -2 * Pr * tau * (R0 * tau - 1)

    p = a1 * a2 / (6 * a3 ** 2) - a0 / (2 * a3) - (a2 / (3 * a3)) ** 3
    q = (a2 / (3 * a3)) ** 2 - a1 / (3 * a3)

    tlam = 2 * np.sqrt(q) * np.cos(np.arccos(p / np.sqrt(q) ** 3) / 3) - a2 / (3 * a3)

    l_f = (Pr * (1 + tlam - R0 * (tau + tlam)) / (R0 * (1 + tlam) * (Pr + tlam) * (tau + tlam))) ** 0.25

    lam_f = l_f ** 2 * tlam

    return lam_f, l_f
    # divide by tau because the above equations are in units of kappa_T, but this project works in kappa_C units
    # return lam_f/tau, l_f


def R_from_r(r, tau):
    return r*(-1 + 1/tau) + 1


def r_from_R(R, tau):
    return (R-1)/(-1 + 1/tau)


def epsilon_from_r(r, tau):
    R0 = R_from_r(r, tau)
    Rayleigh = 1/(R0*tau)
    return Rayleigh-1


def r_from_epsilon(vareps, tau):
    Rayleigh = vareps + 1
    R0 = 1/(Rayleigh*tau)
    return r_from_R(R0, tau)


show = True
save = False
plotname = 'FC'

task = 'FC'
ylabel = r'$\langle w C \rangle$'


dirs = []
runs = []
rs = []
colors = []
linestyles = []

dirs.append('runs/eps0p1_tau5e-7_Pr1e-6/scalars/')
runs.append(r'$\varepsilon = 0.1$')
rs.append(r_from_epsilon(0.1, 5e-7))
colors.append('C0')
linestyles.append('-')

dirs.append('runs/eps10_tau5e-7_Pr1e-6_hres/scalars/')
runs.append(r'$\varepsilon = 10$')
rs.append(r_from_epsilon(10, 5e-7))
colors.append('C1')
linestyles.append('-')

# dirs.append('runs/eps100_tau5e-7_Pr1e-6_vvhres/scalars/')
dirs.append('runs/eps100_tau5e-4_Pr1e-3_vhres/scalars/')
runs.append(r'$\varepsilon = 100$')
rs.append(r_from_epsilon(100, 5e-4))
colors.append('C2')
linestyles.append('-')

dirs.append('runs/eps300_tau5e-7_Pr1e-6_vvhresx1p5/scalars/')
runs.append(r'$\varepsilon = 300$')
rs.append(r_from_epsilon(300, 5e-7))
colors.append('r')
linestyles.append('-')

Pr = 1e-6
tau = 5e-7
R0s = np.array([R_from_r(r, tau) for r in rs])
prefactors = np.ones(len(runs))
# prefactors = np.array([(-1 + 1/(tau*r0))**-2 for r0 in R0s])
# time_rescales = np.ones(len(runs))
time_rescales = np.array([eval_finger(r0, Pr, tau)[0] for r0 in R0s])
tshifts = np.zeros(len(dirs))

# Called Srms because I originally wrote it specifically for plotting S_rms before generalizing it
def get_scalar(fname):
    with h5py.File(fname, 'r') as f:
        sim_time = np.array(f['scales/sim_time'])
        Srms = np.array(f['tasks/{}'.format(task)][:,0,0,0])
    return sim_time, np.abs(Srms)


sim_time_arr = []
Srms_arr = []
for scalars_dir in dirs:
    # check how many scalars_s*.h5 files there are in this directory and stitch them together
    last_ind = max([int(fpath.split('scalars_s')[-1][0]) for fpath in glob.glob(scalars_dir + '*.h5')])
    names = [scalars_dir + 'scalars_s{}.h5'.format(i) for i in range(1, last_ind + 1)]
    for n, name in enumerate(names):
        if n==0:
            sim_time, Srms = get_scalar(name)
        else:
            sim_time2, Srms2 = get_scalar(name)
            if sim_time2[0] < sim_time[-1]:  # check for redundant entries
                extra_ind = np.argmin(np.abs(sim_time - sim_time2[0]))  # index in sim_time that coincides with first redundant datapoint
                sim_time = np.append(sim_time[:extra_ind], sim_time2)
                Srms = np.append(Srms[:extra_ind], Srms2)
            else:
                sim_time = np.append(sim_time, sim_time2)
                Srms = np.append(Srms, Srms2)
    sim_time_arr.append(sim_time)
    Srms_arr.append(Srms)

for i in range(len(sim_time_arr)):
    sim_time_arr[i] = sim_time_arr[i]*time_rescales[i]
# print(time_rescales)

# equations for optimal wavenumber and growth rate of the IFSC model, from Xie et al
# k_opt1 = (0.5*(-2 - Ra + np.sqrt(Ra**2 + 8*Ra)))**(1/4)
# lambda_opt1 = k_opt1**2 * (3*Ra - np.sqrt(Ra**2 + 8*Ra))/(np.sqrt(Ra**2 + 8*Ra) - Ra)



# lambda_opt2, k_opt2 = eval_finger(1/(tau*Ra), Pr, tau)

plt.subplot(2, 1, 1)
for i in range(len(sim_time_arr)):
    plt.semilogy(sim_time_arr[i], Srms_arr[i], label=runs[i], ls=linestyles[i], color=colors[i])
    plt.axvline(sim_time_arr[i][int(3*len(sim_time_arr[i]) / 4)], color=colors[i])
# if compare_lambda:
#     # fit_min = 1e-4
#     # fit_min = 1e-28
#     fit_min = prefactors[i]*Srms_arr[0][2]
#     if squared_quantity:
#         prefac = 2
#     else:
#         prefac = 1
#     plt.semilogy(sim_time_arr[0][:50], fit_min*np.exp(prefac*lambda_opt1*sim_time_arr[0][:50]), ls='--', label=r'$\lambda_\mathrm{IFSC}$')
#     plt.semilogy(sim_time_arr[0][:50], fit_min*np.exp(prefac*lambda_opt2*sim_time_arr[0][:50]), ls='--', label=r'$\lambda_\mathrm{full}$')
plt.xlim(xmin=0)
plt.xlim(xmax=2e-5)
plt.ylabel(ylabel)
# plt.xlabel(r'$t$')
plt.legend(ncol=3)

plt.subplot(2, 1, 2)
for i in range(len(sim_time_arr)):
    plt.plot(sim_time_arr[i], Srms_arr[i], label=runs[i], ls=linestyles[i], color=colors[i])
    plt.axvline(sim_time_arr[i][int(len(sim_time_arr[i])/2)], color=colors[i])
# plt.axhline(Srms_arr[0][-1])
plt.xlim(xmin=0)
plt.xlim(xmax=2e-5)
plt.ylim(ymin=0)
plt.ylabel(ylabel)
plt.xlabel(r'$t$')
# plt.axvline(1.41e4)
# plt.legend()

if save:
    plt.savefig('plots/'+plotname + '.pdf')
    plt.savefig('plots/'+plotname + '.png')
if show:
    plt.show()
