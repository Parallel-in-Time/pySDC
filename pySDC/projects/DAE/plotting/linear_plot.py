import matplotlib.pyplot as plt
import numpy as np
from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier


def linear_plot():
    '''Loads solution data from an .npy file and plots specified parameters with respect to each other on a linear axis'''

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['newton_tol'] = 1e-6  # tollerance for implicit solver
    problem_params['nvars'] = 5

    trans_amp = one_transistor_amplifier(problem_params)

    data = np.load("/Users/heisenberg/Workspace/pySDC/pySDC/projects/DAE/misc/data/one_trans_amp.npy")
    y_data = [trans_amp.u_exact(t)[4] for t in data[:, 0]]
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    # ax.plot(data[:, 1], data[:, 2])
    ax.plot(data[:, 0], data[:, 1], label="x")
    ax.plot(data[:, 0], y_data, label="y_interp")
    ax.plot(data[:, 0], data[:, 5], label="y", linestyle=':')
    # title='Convergence plot two stage implicit Runge-Kutta with Gauss nodes'
    ax.set(xlabel='t', ylabel='size')
    ax.grid(visible=True)
    # fig.tight_layout()
    plt.legend()
    plt.show()
    # plt.savefig('../results/problematic_good.png')


if __name__ == "__main__":
    linear_plot()
