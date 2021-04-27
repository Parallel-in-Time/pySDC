import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh, parallel_imex_mesh
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit, allencahn_semiimplicit


# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


def setup_problem():

    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = (128, 128)
    problem_params['eps'] = 0.04
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-07
    problem_params['lin_tol'] = 1E-08
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25

    return problem_params


def run_implicit_Euler(t0, dt, Tend):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    problem = allencahn_fullyimplicit(problem_params=setup_problem(), dtype_u=parallel_mesh, dtype_f=parallel_mesh)

    u = problem.u_exact(t0)

    radius = []
    exact_radius = []
    nsteps = int((Tend - t0) / dt)
    startt = time.time()
    t = t0
    for n in range(nsteps):

        u_new = problem.solve_system(rhs=u, factor=dt, u0=u, t=t)

        u = u_new
        t += dt

        r, re = compute_radius(u, problem.dx, t, problem.params.radius)
        radius.append(r)
        exact_radius.append(re)

        print(' ... done with time = %6.4f, step = %i / %i' % (t, n + 1, nsteps))

    print('Time to solution: %6.4f sec.' % (time.time() - startt))

    fname = 'data/AC_reference_Tend{:.1e}'.format(Tend) + '.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    err = np.linalg.norm(uref - u, np.inf)
    print('Error vs. reference solution: %6.4e' % err)

    return err, radius, exact_radius


def run_imex_Euler(t0, dt, Tend):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    problem = allencahn_semiimplicit(problem_params=setup_problem(), dtype_u=parallel_mesh, dtype_f=parallel_imex_mesh)

    u = problem.u_exact(t0)

    radius = []
    exact_radius = []
    nsteps = int((Tend - t0) / dt)
    startt = time.time()
    t = t0
    for n in range(nsteps):

        f = problem.eval_f(u, t)
        rhs = u + dt * f.expl
        u_new = problem.solve_system(rhs=rhs, factor=dt, u0=u, t=t)

        u = u_new
        t += dt

        r, re = compute_radius(u, problem.dx, t, problem.params.radius)
        radius.append(r)
        exact_radius.append(re)

        print(' ... done with time = %6.4f, step = %i / %i' % (t, n + 1, nsteps))

    print('Time to solution: %6.4f sec.' % (time.time() - startt))

    fname = 'data/AC_reference_Tend{:.1e}'.format(Tend) + '.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    err = np.linalg.norm(uref - u, np.inf)
    print('Error vs. reference solution: %6.4e' % err)

    return err, radius, exact_radius


def run_CrankNicholson(t0, dt, Tend):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    problem = allencahn_fullyimplicit(problem_params=setup_problem(), dtype_u=parallel_mesh, dtype_f=parallel_mesh)

    u = problem.u_exact(t0)

    radius = []
    exact_radius = []
    nsteps = int((Tend - t0)/dt)
    startt = time.time()
    t = t0
    for n in range(nsteps):

        rhs = u + dt / 2 * problem.eval_f(u, t)
        u_new = problem.solve_system(rhs=rhs, factor=dt / 2, u0=u, t=t)

        u = u_new
        t += dt

        r, re = compute_radius(u, problem.dx, t, problem.params.radius)
        radius.append(r)
        exact_radius.append(re)

        print(' ... done with time = %6.4f, step = %i / %i' % (t, n + 1, nsteps))

    print('Time to solution: %6.4f sec.' % (time.time() - startt))

    fname = 'data/AC_reference_Tend{:.1e}'.format(Tend) + '.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    err = np.linalg.norm(uref - u, np.inf)
    print('Error vs. reference solution: %6.4e' % err)

    return err, radius, exact_radius


def compute_radius(u, dx, t, init_radius):

    c = np.count_nonzero(u >= 0.0)
    radius = np.sqrt(c / np.pi) * dx

    exact_radius = np.sqrt(max(init_radius ** 2 - 2.0 * t, 0))

    return radius, exact_radius


def plot_radius(xcoords, exact_radius, radii):

    fig, ax = plt.subplots()
    plt.plot(xcoords, exact_radius, color='k', linestyle='--', linewidth=1, label='exact')

    for type, radius in radii.items():
        plt.plot(xcoords, radius, linestyle='-', linewidth=2, label=type)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    ax.set_ylabel('radius')
    ax.set_xlabel('time')
    ax.grid()
    ax.legend(loc=3)
    fname = 'data/AC_contracting_circle_standard_integrators'
    plt.savefig('{}.pdf'.format(fname),  bbox_inches='tight')

    # plt.show()


def main_radius(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # setup parameters "in time"
    t0 = 0.0
    dt = 0.001
    Tend = 0.032

    radii = {}
    _, radius, exact_radius = run_implicit_Euler(t0=t0, dt=dt, Tend=Tend)
    radii['implicit-Euler'] = radius
    _, radius, exact_radius = run_imex_Euler(t0=t0, dt=dt, Tend=Tend)
    radii['imex-Euler'] = radius
    _, radius, exact_radius = run_CrankNicholson(t0=t0, dt=dt, Tend=Tend)
    radii['CrankNicholson'] = radius

    xcoords = [t0 + i * dt for i in range(int((Tend - t0) / dt))]
    plot_radius(xcoords, exact_radius, radii)


def main_error(cwd=''):

    t0 = 0
    Tend = 0.032

    errors = {}
    # err, _, _ = run_implicit_Euler(t0=t0, dt=0.001/512, Tend=Tend)
    # errors['implicit-Euler'] = err
    # err, _, _ = run_imex_Euler(t0=t0, dt=0.001/512, Tend=Tend)
    # errors['imex-Euler'] = err
    err, _, _ = run_CrankNicholson(t0=t0, dt=0.001/64, Tend=Tend)
    errors['CrankNicholson'] = err


if __name__ == "__main__":
    main_error()
    # main_radius()
