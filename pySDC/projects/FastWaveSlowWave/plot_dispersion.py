import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sympy
from pylab import rcParams

from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.acoustic_helpers.standard_integrators import dirk, rk_imex

from pySDC.core.Step import step


def findomega(stab_fh):
    assert np.array_equal(np.shape(stab_fh), [2, 2]), 'Not 2x2 matrix...'
    omega = sympy.Symbol('omega')
    func = (sympy.exp(-1j * omega) - stab_fh[0, 0]) * (sympy.exp(-1j * omega) - stab_fh[1, 1]) - \
        stab_fh[0, 1] * stab_fh[1, 0]
    solsym = sympy.solve(func, omega)
    sol0 = complex(solsym[0])
    sol1 = complex(solsym[1])
    if sol0.real >= 0:
        sol = sol0
    elif sol1.real >= 0:
        sol = sol1
    else:
        print("Two roots with real part of same sign...")
        sol = sol0
    return sol


def compute_and_plot_dispersion():
    problem_params = dict()
    # SET VALUE FOR lambda_slow AND VALUES FOR lambda_fast ###
    problem_params['lambda_s'] = np.array([0.0])
    problem_params['lambda_f'] = np.array([0.0])
    problem_params['u0'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    # SET TYPE AND NUMBER OF QUADRATURE NODES ###
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['do_coll_update'] = True
    sweeper_params['num_nodes'] = 3

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = swfw_scalar  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = dict()  # pass step parameters

    # SET NUMBER OF ITERATIONS ###
    K = 3

    # ORDER OF DIRK/IMEX IS EQUAL TO NUMBER OF ITERATIONS AND THUS ORDER OF SDC ###
    dirk_order = K

    c_speed = 1.0
    U_speed = 0.05

    # now the description contains more or less everything we need to create a step
    S = step(description=description)

    L = S.levels[0]

    # u0 = S.levels[0].prob.u_exact(t0)
    # S.init_step(u0)
    QE = L.sweep.QE[1:, 1:]
    QI = L.sweep.QI[1:, 1:]
    Q = L.sweep.coll.Qmat[1:, 1:]
    nnodes = L.sweep.coll.num_nodes
    dt = L.params.dt

    Nsamples = 15
    k_vec = np.linspace(0, np.pi, Nsamples + 1, endpoint=False)
    k_vec = k_vec[1:]
    phase = np.zeros((3, Nsamples))
    amp_factor = np.zeros((3, Nsamples))

    for i in range(0, np.size(k_vec)):

        Cs = -1j * k_vec[i] * np.array([[0.0, c_speed], [c_speed, 0.0]], dtype='complex')
        Uadv = -1j * k_vec[i] * np.array([[U_speed, 0.0], [0.0, U_speed]], dtype='complex')

        LHS = np.eye(2 * nnodes) - dt * (np.kron(QI, Cs) + np.kron(QE, Uadv))
        RHS = dt * (np.kron(Q, Uadv + Cs) - np.kron(QI, Cs) - np.kron(QE, Uadv))

        LHSinv = np.linalg.inv(LHS)
        Mat_sweep = np.linalg.matrix_power(LHSinv.dot(RHS), K)
        for k in range(0, K):
            Mat_sweep = Mat_sweep + np.linalg.matrix_power(LHSinv.dot(RHS), k).dot(LHSinv)
        ##
        # ---> The update formula for this case need verification!!
        update = dt * np.kron(L.sweep.coll.weights, Uadv + Cs)

        y1 = np.array([1, 0], dtype='complex')
        y2 = np.array([0, 1], dtype='complex')
        e1 = np.kron(np.ones(nnodes), y1)
        stab_fh_1 = y1 + update.dot(Mat_sweep.dot(e1))
        e2 = np.kron(np.ones(nnodes), y2)
        stab_fh_2 = y2 + update.dot(Mat_sweep.dot(e2))
        stab_sdc = np.column_stack((stab_fh_1, stab_fh_2))

        # Stability function of backward Euler is 1/(1-z); system is y' = (Cs+Uadv)*y
        # stab_ie = np.linalg.inv( np.eye(2) - step.status.dt*(Cs+Uadv) )

        # For testing, insert exact stability function exp(-dt*i*k*(Cs+Uadv)
        # stab_fh = la.expm(Cs+Uadv)

        dirkts = dirk(Cs + Uadv, dirk_order)
        stab_fh1 = dirkts.timestep(y1, 1.0)
        stab_fh2 = dirkts.timestep(y2, 1.0)
        stab_dirk = np.column_stack((stab_fh1, stab_fh2))

        rkimex = rk_imex(M_fast=Cs, M_slow=Uadv, order=K)
        stab_fh1 = rkimex.timestep(y1, 1.0)
        stab_fh2 = rkimex.timestep(y2, 1.0)
        stab_rk_imex = np.column_stack((stab_fh1, stab_fh2))

        sol_sdc = findomega(stab_sdc)
        sol_dirk = findomega(stab_dirk)
        sol_rk_imex = findomega(stab_rk_imex)

        # Now solve for discrete phase
        phase[0, i] = sol_sdc.real / k_vec[i]
        amp_factor[0, i] = np.exp(sol_sdc.imag)
        phase[1, i] = sol_dirk.real / k_vec[i]
        amp_factor[1, i] = np.exp(sol_dirk.imag)
        phase[2, i] = sol_rk_imex.real / k_vec[i]
        amp_factor[2, i] = np.exp(sol_rk_imex.imag)

    rcParams['figure.figsize'] = 1.5, 1.5
    fs = 8
    fig = plt.figure()
    plt.plot(k_vec, (U_speed + c_speed) + np.zeros(np.size(k_vec)), '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, phase[1, :], '-', color='g', linewidth=1.5, label='DIRK(' + str(dirkts.order) + ')')
    plt.plot(k_vec, phase[2, :], '-+', color='r', linewidth=1.5, label='IMEX(' + str(rkimex.order) + ')',
             markevery=(2, 3), mew=1.0)
    plt.plot(k_vec, phase[0, :], '-o', color='b', linewidth=1.5, label='SDC(' + str(K) + ')', markevery=(1, 3),
             markersize=fs / 2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([0.0, 1.1 * (U_speed + c_speed)])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs - 2})
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    filename = 'data/phase-K' + str(K) + '-M' + str(sweeper_params['num_nodes']) + '.png'
    plt.gcf().savefig(filename, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(k_vec, 1.0 + np.zeros(np.size(k_vec)), '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, amp_factor[1, :], '-', color='g', linewidth=1.5, label='DIRK(' + str(dirkts.order) + ')')
    plt.plot(k_vec, amp_factor[2, :], '-+', color='r', linewidth=1.5, label='IMEX(' + str(rkimex.order) + ')',
             markevery=(2, 3), mew=1.0)
    plt.plot(k_vec, amp_factor[0, :], '-o', color='b', linewidth=1.5, label='SDC(' + str(K) + ')', markevery=(1, 3),
             markersize=fs / 2)
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([k_vec[0], k_vec[-1:]])
    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs - 2})
    plt.gca().set_ylim([0.0, 1.1])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    filename = 'data/ampfactor-K' + str(K) + '-M' + str(sweeper_params['num_nodes']) + '.png'
    plt.gcf().savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    compute_and_plot_dispersion()
