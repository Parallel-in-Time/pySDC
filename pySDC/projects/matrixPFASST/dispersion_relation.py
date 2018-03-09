import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.datatype_classes.complex_mesh import mesh as cmesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh as mesh_to_mesh_nocoarse
from pySDC.projects.matrixPFASST.allinclusive_matrix_nonMPI import allinclusive_matrix_nonMPI


# computes the frequency omega = 1j*log(R)
def solve_omega(R):
    return 1j * (np.log(abs(R)) + 1j * np.angle(R))


# finds all roots of the stability function
def findroots(R, n):
    assert abs(n - float(int(n))) < 1e-14, "n must be an integer or a float equal to an integer"
    p = np.zeros(int(n) + 1, dtype='complex')
    p[-1] = -R
    p[0] = 1.0
    return np.roots(p)


# normalises the stability function of Parareal from [0,Tend] to [0,1]
# by computing all Tend=P roots and then selecting the one that is closest to the given
# target angle
def normalise(R, T, target):
    roots = findroots(R, T)

    # make sure all computed values are actually roots
    for x in roots:
        assert abs(x ** T - R) < 1e-3, ("Element in roots not a proper root: err=%5.3e" % abs(x ** T - R))

    # find root that minimises distance to target angle
    minind = np.argmin(abs(np.angle(roots) - target))
    return roots[minind]


def scalar_equation_setup():
    """
    Setup routine for the test equation

    Args:
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0.0
    level_params['dt'] = 1.0
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3, 2]
    sweeper_params['QI'] = 'LU'
    sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['u0'] = 1.0  # initial value (for all instances)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = testequation0d  # pass problem class
    description['problem_params'] = problem_params
    description['dtype_u'] = cmesh  # pass data type for u
    description['dtype_f'] = cmesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_nocoarse  # pass spatial transfer class
    description['space_transfer_params'] = dict()  # pass paramters for spatial transfer

    return description, controller_params


def compute_dispersion_relation(npoints=None, f=None):

    num_procs = 16
    Tend = float(num_procs)

    deltas = -1j * np.linspace(0, np.pi, npoints + 1, endpoint=False)[1:]
    niters = [1, 2, 3]

    phase = np.zeros((len(niters), npoints), dtype=np.complex64)
    amp_factor = np.zeros((len(niters), npoints), dtype=np.complex64)


    for idx, niter in enumerate(niters):

        target_angle = np.angle(np.exp(deltas[0]))

        for idy, delta in enumerate(deltas):

            description, controller_params = scalar_equation_setup()
            description['problem_params']['lambdas'] = [[delta]]  # pass problem parameters
            description['step_params']['maxiter'] = niter

            # instantiate controller
            controller = allinclusive_matrix_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                                    description=description)

            # build propagation matrix using the prescribed number of iterations (or any other, if needed)
            prop = controller.build_propagation_matrix(niter=description['step_params']['maxiter'])

            pfasst_norm = normalise(prop[0, 0], Tend, target_angle)

            # Make sure that stab_norm*dt = stab
            err = abs(pfasst_norm ** Tend - prop[0, 0])
            if err > 1e-10:
                print("WARNING: power of norm. update does not match update over full length of time. error %5.3e" % err)

            target_angle = np.angle(pfasst_norm)

            sol_pfasst = solve_omega(pfasst_norm)
            phase[idx, idy] = sol_pfasst.real / -delta.imag
            amp_factor[idx, idy] = np.exp(sol_pfasst.imag)

    return niters, deltas, phase, amp_factor


def plot_dispersion_relations(niters, deltas, phase, amp_factor):

    colors = ['r', 'b', 'g']
    rcParams['figure.figsize'] = 3.54, 3.54
    fs = 8
    fig = plt.figure()
    for idx, niter in enumerate(niters):
        plt.plot(-deltas.imag, phase[idx, :], '--', color=colors[idx], linewidth=1.5, label='PFASST, k=%i' % niter)
    plt.plot(-deltas.imag, 0 * deltas + 1, '-', color='k', linewidth=0.5, label='1')
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([-deltas[0].imag, -deltas[-1].imag])
    plt.ylim([0.0, 1.2])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs - 2})
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    # plt.show()
    filename = 'pfasst-dispersion-phase.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    # call(["pdfcrop", filename, filename])

    fig = plt.figure()
    for idx, niter in enumerate(niters):
        plt.plot(-deltas.imag, amp_factor[idx, :], '--', color=colors[idx], linewidth=1.5, label='PFASST, k=%i' % niter)
    plt.plot(-deltas.imag, 0 * deltas + 1, '-', color='k', linewidth=0.5, label='1')
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([-deltas[0].imag, -deltas[-1].imag])
    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs - 2})
    plt.gca().set_ylim([0.0, 1.2])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    # plt.show()
    filename = 'pfasst-dispersion-ampf.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    # call(["pdfcrop", filename, filename])
#

def main():

    f = open('dispersion_relation.txt', 'w')
    niters, deltas, phase, amp_factor = compute_dispersion_relation(npoints=32, f=f)
    plot_dispersion_relations(niters, deltas, phase, amp_factor)
    f.close()


if __name__ == "__main__":
    main()
