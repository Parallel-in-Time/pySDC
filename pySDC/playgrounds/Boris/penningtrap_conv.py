import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np


from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.playgrounds.Boris.penningtrap_HookClass import convergence_data


def error_calculator(u_ex, u):
    return np.linalg.norm(u_ex - u, np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)


def compute_covnergence_data(cwd=""):
    """
    Routine to run the penning trap example with different orders

    Args:
        cwd (string): current working directory
    """

    num_procs = 1

    # initialize level parameters
    level_params = dict()
    level_params["restol"] = 1e-16

    # This comes as read-in for the problem params
    step_params = dict()

    # This comes as read-in for the problem params
    problem_params = dict()
    problem_params["omega_E"] = 4.9
    problem_params["omega_B"] = 25.0
    problem_params["u0"] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]])
    problem_params["nparts"] = 1
    problem_params["sig"] = 0.1
    problem_params["Tend"] = 128 * 0.015625

    # This comes as read-in for the sweeper params
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params["num_nodes"] = 3
    sweeper_params["do_coll_update"] = True
    sweeper_params["initial_guess"] = "random"

    # initialize controller parameters
    controller_params = dict()
    controller_params["logger_level"] = 30
    controller_params["hook_class"] = convergence_data

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description["problem_params"] = problem_params
    description["problem_class"] = penningtrap
    description["sweeper_class"] = boris_2nd_order
    description["sweeper_params"] = sweeper_params

    Miter = [1, 2, 3]
    t0 = 0.0
    Tend = 128 * 0.015625
    titer = 3
    values = ["position", "velocity"]

    error = dict()
    for order in Miter:

        error_val = dict()
        u_val = dict()
        uex_val = dict()

        for ii, jj in enumerate(values):
            error_val[jj] = np.zeros([3, titer])

        step_params["maxiter"] = order
        description["step_params"] = step_params

        for ii in range(titer):

            dt = 0.015625 / 2**ii

            level_params["dt"] = dt
            description["level_params"] = level_params

            # instantiate the controller
            controller = controller_nonMPI(
                num_procs=num_procs,
                controller_params=controller_params,
                description=description,
            )

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done ...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # extract values from stats
            extract_stats = filter_stats(stats, type="error")
            sortedlist_stats = sort_stats(extract_stats, sortby="time")

            sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
            sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel
            # sort values and compute error
            for mm, nn in enumerate(values):
                data = sortedlist_stats[0][1][nn].values()
                u_val[nn] = np.array(list(data))
                u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                data = sortedlist_stats[0][1][nn + "_exact"].values()
                uex_val[nn] = np.array(list(data))
                uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                error_val[nn][:, ii] = error_calculator(uex_val[nn], u_val[nn])

        error[order] = error_val
    # save all values for the plot the graph
    time = [dt * 2**i for i in range(titer)]
    np.save("data/conv-data.npy", error)
    np.save("data/time.npy", np.flip(time))


def plot_convergence(cwd=""):
    """
    Plotting routine for the convergence data

    Args:
        cwd (string): current working directory
    """
    # needs to include the values for the plot the convergence
    fs = 10
    Kiter = np.array([1, 2, 3])
    omega_B = 25.0
    num_nodes = 3
    plot = "position"
    axis = 1
    order = np.min([Kiter, np.ones(len(Kiter)) * 2 * num_nodes], 0)

    time = np.load("data/time.npy")
    error = np.load("data/conv-data.npy", allow_pickle=True).item()

    color = ["r", "blue", "g"]
    shape = ["o", "d", "s"]

    for ii, jj in enumerate(Kiter):
        plt.loglog(
            time * omega_B,
            error[jj][plot][axis, :],
            color=color[ii],
            marker=shape[ii],
            ls="",
            ms=fs,
            label="k={}".format(jj),
        )
        plt.loglog(
            time * omega_B,
            error[jj][plot][axis, 0] * (time / time[0]) ** (order[ii]),
            color="black",
        )
        plt.text(
            time[1] * omega_B,
            0.3 * error[jj][plot][axis, 0] * (time[1] / time[0]) ** (order[ii]),
            r"$\mathcal{O}(\Delta t^{%d})$" % (order[ii]),
            size=2 * fs,
        )
    plt.grid(True)
    # title for the plot
    plt.title(f"$x_{axis+1}$ coordinate, M={num_nodes}")
    plt.xlabel(r"$\omega_{B}\cdot \Delta t$")
    plt.ylabel("Relative error")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("conv_{}_M={}.png".format(axis, num_nodes))
    plt.show()


if __name__ == "__main__":
    compute_covnergence_data()
    plot_convergence()
