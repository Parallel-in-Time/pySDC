import matplotlib

matplotlib.use("agg")

import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.Errors import ProblemError

from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.core.Step import step


def compute_stability():
    """
    Runtine to compute modulues of the stability function

    Args:
        None


    Returns:
        numpy.narray: values for the spring pendulum
        numpy.narray: values for the Friction
        numpy.narray: number of num_nodes
        numpy.narray: number of iterations
        numpy.narray: moduli for the SDC
        numpy.narray: moduli for the K_{sdc} marrix
        numpy.narray: moduli for the Pircard iteration
        numpy.narray: moduli for the K_{sdc} Picard iteration
    """
    N_k = 400
    N_mu = 400

    k_max = 20.0
    mu_max = 20.0
    lambda_k = np.linspace(0.0, k_max, N_k)
    lambda_mu = np.linspace(0.0, mu_max, N_mu)

    problem_params = dict()
    # set value for k and mu
    problem_params["k"] = 0
    problem_params["mu"] = 0
    problem_params["u0"] = np.array([1, 1])

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["collocation_class"] = CollGaussLegendre
    sweeper_params["num_nodes"] = 2
    sweeper_params["do_coll_update"] = True
    sweeper_params["picard_mats_sweep"] = True

    # initialize level parameters
    level_params = dict()
    level_params["dt"] = 1.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description["problem_class"] = harmonic_oscillator
    description["problem_params"] = problem_params
    description["sweeper_class"] = boris_2nd_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = dict()

    K = 100

    S = step(description=description)

    L = S.levels[0]

    Q = L.sweep.coll.Qmat[1:, 1:]
    QQ = np.dot(Q, Q)
    nnodes = L.sweep.coll.num_nodes
    dt = L.params.dt

    Q_coll = np.block([[QQ, np.zeros([nnodes, nnodes])], [np.zeros([nnodes, nnodes]), Q]])
    qQ = np.dot(L.sweep.coll.weights, Q)

    ones = np.block([[np.ones(nnodes), np.zeros(nnodes)], [np.zeros(nnodes), np.ones(nnodes)]])

    q_mat = np.block(
        [
            [dt**2 * qQ, np.zeros(nnodes)],
            [np.zeros(nnodes), dt * L.sweep.coll.weights],
        ]
    )

    stab = np.zeros((N_k, N_mu), dtype="complex")
    stab_picard = np.zeros((N_k, N_mu))
    Kstab = np.zeros((N_k, N_mu))
    KPstab = np.zeros((N_k, N_mu))
    for i in range(0, N_k):
        for j in range(0, N_mu):
            k = lambda_k[i]
            mu = lambda_mu[j]
            F = np.block(
                [
                    [-k * np.eye(nnodes), -mu * np.eye(nnodes)],
                    [-k * np.eye(nnodes), -mu * np.eye(nnodes)],
                ]
            )
            if K != 0:
                lambdas = [k, mu]
                Mat_sweep, Keig = L.sweep.get_scalar_problems_manysweep_mats(nsweeps=K, lambdas=lambdas)
                if L.sweep.params.picard_mats_sweep:
                    (
                        Picard_mats_sweep,
                        Kpicard,
                    ) = L.sweep.get_scalar_problems_picardsweep_mats(nsweeps=K, lambdas=lambdas)
                else:
                    pass
                    ProblemError("Picard interation is False")
                Kstab[i, j] = Keig
                if L.sweep.params.picard_mats_sweep:
                    KPstab[i, j] = Kpicard

            else:

                Mat_sweep = np.linalg.inv(np.eye(2 * nnodes) - dt * np.dot(Q_coll, F))

            if L.sweep.params.do_coll_update:
                FP = np.dot(F, Mat_sweep)
                R_mat = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FP) @ ones.T
                stab_fh, v = np.linalg.eig(R_mat)

                if L.sweep.params.picard_mats_sweep:
                    FPicard = np.dot(F, Picard_mats_sweep)
                    R_mat_picard = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FPicard) @ ones.T
                    stab_fh_picard, v = np.linalg.eig(R_mat_picard)
            else:
                pass
                raise ProblemError("Collocation update step is only works for True")

            stab[i, j] = np.max(np.abs(stab_fh))
            if L.sweep.params.picard_mats_sweep:
                stab_picard[i, j] = np.max(np.abs(stab_fh_picard))

    return (
        lambda_k,
        lambda_mu,
        sweeper_params["num_nodes"],
        K,
        dt * stab.real,
        dt * Kstab.real,
        dt * stab_picard.real,
        dt * KPstab.real,
    )


def plot_stability(lambda_k, lambda_mu, num_nodes, K, stab, title):
    """
    Plotting runtine for moduli

    Args:
        spring coefficient: lambda_k
        friction coefficient: lambda_k
        num_nodes (numpy.ndarray): number of nodes
        K (numpy.ndarray): number of iterations
        stabval (numpy.ndarray): moduli
        title: title for the plot
    """

    lam_k_max = np.amax(lambda_k)
    lam_mu_max = np.amax(lambda_mu)

    fs = 12

    levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

    CS1 = plt.contour(lambda_k, lambda_mu, np.absolute(stab.T), levels, linestyles="dashed")
    CS2 = plt.contour(lambda_k, lambda_mu, np.absolute(stab.T), [1.0])

    plt.clabel(CS1, inline=True, fmt="%3.2f", fontsize=fs - 2)
    manual_locations = [(1.5, 2.5)]
    if K > 0:  # for K=0 and no 1.0 isoline, this crashes Matplotlib for somer reason
        plt.clabel(CS2, inline=True, fmt="%3.2f", fontsize=fs - 2, manual=manual_locations)

    plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 1))
    plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 2, 2))
    plt.gca().tick_params(axis="both", which="both", labelsize=fs)
    plt.xlim([0.0, lam_k_max])
    plt.ylim([0.0, lam_mu_max])
    plt.xlabel(r"$\Delta t\cdot \kappa \ (Spring \ pendulum)$", fontsize=fs, labelpad=0.0)
    plt.ylabel(r"$\Delta t\cdot \mu \ (Friction)$", fontsize=fs, labelpad=0.0)
    plt.title("{}  M={} K={}".format(title, num_nodes, K), fontsize=fs)
    # filename = "stability-K" + str(K) + "-M" + str(num_nodes) + title + ".png"
    # fig.savefig(filename, bbox_inches="tight")


def plot_K_sdc(lambda_k, lambda_mu, num_nodes, K, stab, title):
    """
    Plotting runtine for moduli

    Args:
        spring coefficient: lambda_k
        friction coefficient: lambda_k
        num_nodes (numpy.ndarray): number of nodes
        K (numpy.ndarray): number of iterations
        stabval (numpy.ndarray): moduli
        title: title for the plot
    """

    lam_k_max = np.amax(lambda_k)
    lam_mu_max = np.amax(lambda_mu)

    fs = 12

    levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

    CS1 = plt.contour(lambda_k, lambda_mu, np.absolute(stab.T), levels, linestyles="dashed")
    CS2 = plt.contour(lambda_k, lambda_mu, np.absolute(stab.T), [1.0])

    plt.clabel(CS1, inline=True, fmt="%3.2f", fontsize=fs - 2)
    manual_locations = [(1.5, 2.5)]
    if K > 0:  # for K=0 and no 1.0 isoline, this crashes Matplotlib for somer reason
        plt.clabel(CS2, inline=True, fmt="%3.2f", fontsize=fs - 2, manual=manual_locations)

    plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 1))
    plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 2, 2))
    plt.gca().tick_params(axis="both", which="both", labelsize=fs)
    plt.xlim([0.0, lam_k_max])
    plt.ylim([0.0, lam_mu_max])
    plt.xlabel(r"$\Delta t\cdot \kappa \ (Spring \ pendulum)$", fontsize=fs, labelpad=0.0)
    plt.ylabel(r"$\Delta t\cdot \mu \ (Friction)$", fontsize=fs, labelpad=0.0)
    plt.title("{}  M={}".format(title, num_nodes), fontsize=fs)
    # filename = "stability-K" + str(K) + "-M" + str(num_nodes) + title + ".png"
    # fig.savefig(filename, bbox_inches="tight")


def main():
    (
        lambda_k,
        lambda_mu,
        num_nodes,
        K,
        stab,
        Kstab,
        stab_picard,
        KPstab,
    ) = compute_stability()

    plot_stability(lambda_k, lambda_mu, num_nodes, K, stab, "SDC stability")
    plot_K_sdc(lambda_k, lambda_mu, num_nodes, K, Kstab, r"$K_{sdc}$ matrix eigenvalue")
    plot_stability(lambda_k, lambda_mu, num_nodes, K, stab_picard, "Picard stability")
    plot_K_sdc(lambda_k, lambda_mu, num_nodes, K, KPstab, "Picard iteration matrix eigenvalue")


if __name__ == "__main__":
    main()
