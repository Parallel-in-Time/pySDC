import numpy as np
import matplotlib.pyplot as plt
from pySDC.core.Errors import ProblemError
from pySDC.core.Step import step

from pySDC.projects.Second_orderSDC.plot_helper import set_fixed_plot_params
from tabulate import tabulate
class StabilityImplementation:
    """
    Routine to compute the stability domains of different configurations of SDC
    """

    def __init__(self, description, kappa_max=20, mu_max=20, Num_iter=(400, 400), cwd=''):
        self.description = description
        self.kappa_max = kappa_max
        self.mu_max = mu_max
        self.kappa_iter = Num_iter[0]
        self.mu_iter = Num_iter[1]
        self.lambda_kappa = np.linspace(0.0, self.kappa_max, self.kappa_iter)
        self.lambda_mu = np.linspace(1e-10, self.mu_max, self.mu_iter)

        self.K_iter = description['step_params']['maxiter']
        self.num_nodes = description['sweeper_params']['num_nodes']
        self.dt = description['level_params']['dt']
        self.SDC, self.Ksdc, self.picard, self.Kpicard = self.stability_data()
        self.cwd = cwd

    def stability_data(self):
        """
        Computes stability domain matrix for the Harmonic oscillator problem
        Returns:
            numpy.ndarray: domain_SDC
            numpy.ndarray: domain_Ksdc
            numpy.ndarray: domain_picard
            numpy.ndarray: domain_Kpicard
        """
        S = step(description=self.description)

        L = S.levels[0]

        Q = L.sweep.coll.Qmat[1:, 1:]
        QQ = np.dot(Q, Q)
        num_nodes = L.sweep.coll.num_nodes
        dt = L.params.dt
        Q_coll = np.block([[QQ, np.zeros([num_nodes, num_nodes])], [np.zeros([num_nodes, num_nodes]), Q]])
        qQ = np.dot(L.sweep.coll.weights, Q)

        ones = np.block([[np.ones(num_nodes), np.zeros(num_nodes)], [np.zeros(num_nodes), np.ones(num_nodes)]])

        q_mat = np.block(
            [
                [dt**2 * qQ, np.zeros(num_nodes)],
                [np.zeros(num_nodes), dt * L.sweep.coll.weights],
            ]
        )

        domain_SDC = np.zeros((self.kappa_iter, self.mu_iter), dtype="complex")
        domain_picard = np.zeros((self.kappa_iter, self.mu_iter))
        domain_Ksdc = np.zeros((self.kappa_iter, self.mu_iter))
        domain_Kpicard = np.zeros((self.kappa_iter, self.mu_iter))
        for i in range(0, self.kappa_iter):
            for j in range(0, self.mu_iter):
                k = self.lambda_kappa[i]
                mu = self.lambda_mu[j]
                F = np.block(
                    [
                        [-k * np.eye(num_nodes), -mu * np.eye(num_nodes)],
                        [-k * np.eye(num_nodes), -mu * np.eye(num_nodes)],
                    ]
                )
                if self.K_iter != 0:
                    lambdas = [k, mu]
                    SDC_mat_sweep, Ksdc_eigval = L.sweep.get_scalar_problems_manysweep_mats(
                        nsweeps=self.K_iter, lambdas=lambdas
                    )
                    if L.sweep.params.picard_mats_sweep:
                        (
                            Picard_mat_sweep,
                            Kpicard_eigval,
                        ) = L.sweep.get_scalar_problems_picardsweep_mats(nsweeps=self.K_iter, lambdas=lambdas)
                    else:
                        ProblemError("Picard interation is False")
                    domain_Ksdc[i, j] = Ksdc_eigval
                    if L.sweep.params.picard_mats_sweep:
                        domain_Kpicard[i, j] = Kpicard_eigval

                else:
                    SDC_mat_sweep = np.linalg.inv(np.eye(2 * num_nodes) - dt * np.dot(Q_coll, F))

                if L.sweep.params.do_coll_update:
                    FSDC = np.dot(F, SDC_mat_sweep)
                    Rsdc_mat = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FSDC) @ ones.T
                    stab_func, v = np.linalg.eig(Rsdc_mat)

                    if L.sweep.params.picard_mats_sweep:
                        FPicard = np.dot(F, Picard_mat_sweep)
                        Rpicard_mat = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FPicard) @ ones.T
                        stab_func_picard, v = np.linalg.eig(Rpicard_mat)
                else:
                    pass
                    raise ProblemError("Collocation update step is only works for True")

                domain_SDC[i, j] = np.max(np.abs(stab_func))
                if L.sweep.params.picard_mats_sweep:
                    domain_picard[i, j] = np.max(np.abs(stab_func_picard))

        return (
            dt * domain_SDC.real,
            dt * domain_Ksdc.real,
            dt * domain_picard.real,
            dt * domain_Kpicard.real,
        )

    def stability_function_RKN(self, k, mu, dt):
        """
        Stability function of RKN method

        Returns:
            float: maximum absolute values of eigvales
        """
        A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        B = np.array([[0, 0, 0, 0], [0.125, 0, 0, 0], [0.125, 0, 0, 0], [0, 0, 0.5, 0]])
        c = np.array([0, 0.5, 0.5, 1])
        b = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
        bA = np.array([1 / 6, 1 / 6, 1 / 6, 0])
        L = np.eye(4) + k * (dt**2) * B + mu * dt * A
        R = np.block([[-k * np.ones(4)], [-(k * c + mu * np.ones(4))]])

        K = np.linalg.inv(L) @ R.T
        C = np.block([[dt**2 * bA], [dt * b]])
        Y = np.array([[1, dt], [0, 1]]) + C @ K
        eigval = np.linalg.eigvals(Y)

        return np.max(np.abs(eigval))

    def stability_data_RKN(self):
        """
        Compute and store values into a matrix

        Returns:
            numpy.ndarray: stab_RKN
        """
        stab_RKN = np.zeros([self.kappa_iter, self.mu_iter])
        for ii, kk in enumerate(self.lambda_kappa):
            for jj, mm in enumerate(self.lambda_mu):
                stab_RKN[jj, ii] = self.stability_function_RKN(kk, mm, self.dt)

        return stab_RKN

    def plot_stability(self, region, title=""):  # pragma: no cover
        """
        Plotting runtine for moduli

        Args:
            stabval (numpy.ndarray): moduli
            title: title for the plot
        """
        set_fixed_plot_params()
        lam_k_max = np.amax(self.lambda_kappa)
        lam_mu_max = np.amax(self.lambda_mu)

        plt.figure()
        levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

        CS1 = plt.contour(self.lambda_kappa, self.lambda_mu, region.T, levels, colors='k', linestyles="dashed")
        # CS2 = plt.contour(self.lambda_k, self.lambda_mu, np.absolute(region.T), [1.0], colors='r')

        plt.clabel(CS1, inline=True, fmt="%3.2f")

        plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 3, 3))
        plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 3, 3))
        plt.gca().tick_params(axis="both", which="both")
        plt.xlim([0.0, lam_k_max])
        plt.ylim([0.0, lam_mu_max])

        plt.xlabel(r"$\Delta t\cdot \kappa$", labelpad=0.0)
        plt.ylabel(r"$\Delta t\cdot \mu$", labelpad=0.0)
        if self.RKN:
            plt.title(f"{title}")
        if self.radius:
            plt.title("{}  $M={}$".format(title, self.num_nodes))
        else:
            plt.title(r"{}  $M={},\  K={}$".format(title, self.num_nodes, self.K_iter))
        plt.tight_layout()
        plt.savefig(self.cwd + "data/M={}_K={}_redion_{}.pdf".format(self.num_nodes, self.K_iter, title))

    def run_SDC_stability(self):  # pragma: no cover
        self.RKN = False
        self.radius = False
        self.plot_stability(self.SDC, title="SDC stability region")

    def run_Picard_stability(self):  # pragma: no cover
        self.RKN = False
        self.radius = False
        self.plot_stability(self.picard, title="Picard stability region")

    def run_Ksdc(self):  # pragma: no cover
        self.radius = True
        self.plot_stability(self.Ksdc, title="$K_{sdc}$ spectral radius")

    def run_Kpicard(self):  # pragma: no cover
        self.radius = True
        self.plot_stability(self.Kpicard, title="$K_{picard}$ spectral radius")

    def run_RKN_stability(self):  # pragma: no cover
        self.RKN = True
        self.radius = False
        region_RKN = self.stability_data_RKN()
        self.plot_stability(region_RKN.T, title='RKN-4 stability region')


def compute_and_check_stability(description,
    point, compute_interval=False, save_interval_file=False, interval_filename='./data/stab_interval.txt',
    check_stability=False, store_results=False, results_filename='./data/stab_results.txt', quadrature_list=['GAUSS', 'LOBATTO']
):

    # Storage for stability interval and stability check
    interval_data = []
    results_data = []

    # Loop through different numbers of nodes and maximum iterations
    for num_nodes in description['helper_params']['num_nodes_list']:
        for max_iter in description['helper_params']['max_iter_list']:
            for quad_type in description['helper_params']['quad_type_list']:
                # Update simulation parameters
                description['sweeper_params']['num_nodes'] = num_nodes
                description['sweeper_params']['quad_type'] = quad_type
                description['step_params']['maxiter'] = max_iter

                # Create Stability_implementation instance for stability check

                stab_model = StabilityImplementation(description, kappa_max=point[0], mu_max=point[1], Num_iter=description['helper_params']['Num_iter'])
                if compute_interval:


                    # Extract the values where SDC is less than or equal to 1
                    mask = stab_model.picard <= 1 + 1e-14
                    for ii in range(len(mask)):

                        if mask[ii]:
                            kappa_max_interval = stab_model.lambda_kappa[ii]
                        else:
                            break

                    # Add row to the interval data
                    interval_data.append([quad_type, num_nodes, max_iter, kappa_max_interval])

                if check_stability:


                    # Check stability and print results
                    if stab_model.SDC[-1, -1] <= 1:
                        stability_result = "Stable"
                    else:
                        stability_result = "Unstable. Increase M or K"

                    # Add row to the results data
                    results_data.append([
                        quad_type, num_nodes, max_iter, point, stability_result, stab_model.SDC[-1, -1]
                    ])

    # Define column names for interval data
    interval_headers = ["Quad Type", "Num Nodes", "Max Iter", 'kappa_max']

    # Define column names for results data
    results_headers = ["Quad Type", "Num Nodes", "Max Iter", "(kappa, mu)", "Stability", "Spectral Radius"]
    print(results_data)
    # Print or save the tables using tabulate
    if save_interval_file and compute_interval:
        interval_table_str = tabulate(interval_data, headers=interval_headers, tablefmt="grid")
        with open(interval_filename, 'w') as file:
            file.write(interval_table_str)
        print(f"Stability Interval Table saved to {interval_filename}")

    if store_results and check_stability:
        results_table_str = tabulate(results_data, headers=results_headers, tablefmt="grid")
        with open(results_filename, 'w') as file:
            file.write(results_table_str)
        print(f"Stability Results Table saved to {results_filename}")

    if compute_interval:
        print("Stability Interval Table:")
        print(tabulate(interval_data, headers=interval_headers, tablefmt="grid"))

    if check_stability:
        print("\nStability Results Table:")
        print(tabulate(results_data, headers=results_headers, tablefmt="grid"))
