import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability, plot_contraction
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.heat import run_heat
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.FDeigenvalues import get_finite_difference_eigenvalues
from pySDC.helpers.stats_helper import get_sorted
from pySDC.playgrounds.Preconditioners.optimize_diagonal_preconditioner import single_run
from pySDC.playgrounds.Preconditioners.configs import (
    load_precon,
    colors,
    get_params,
    get_params_for_stiffness_plot,
)
from pySDC.playgrounds.Preconditioners.hooks import log_error_at_iterations


class PreconPostProcessing:
    def __init__(
        self, problem, nodes, label='', color=None, ls='-', source='optimization', parallelizable=False, **kwargs
    ):
        '''
        Load a preconditioner for postprocessing

        Args:
            problem (str): The name of the problem that has been run to obtain the preconditioner
            nodes (int): Number of nodes that have been used
            label (str): The label that will be displayed in plots for this preconditioner
            color (str): The color of lines associated with this preconditioner
            ls (str): The linesyle for plotting this preconditioner
        '''
        # load data
        self.data = load_precon(problem, nodes, **kwargs)
        self.sweeper_params = self.data['sweeper_params']

        # plotting stuff
        self.label = label
        self.color = color
        self.ls = ls

        # meta data
        self.source = source
        self.parallelizable = parallelizable
        self.normalized = self.data['normalized']
        self.semi_diagonal = self.data['use_first_row']
        self.nodes = nodes
        self.random_ig = self.data['random_IG'] and not self.source == 'optimization'

        # prepare empty variables
        self.dahlquist_stats = None
        self.re_range = np.linspace(-30, 30, 400)
        self.im_range = np.linspace(-50, 50, 400)

    def change_dahlquist_range(self, re=None, im=None, res=400, log=False, **kwargs):
        '''
        Change the range in the complex plane for running the Dahlquist problems and rerun them

        Args:
            re (tuple): Range for the real part
            im (tuple): Range for the imaginary part
            res (int): Number of entries per direction

        Returns:
            None
        '''

        def get_range(start, end, res, log):
            if log:
                return np.append(-np.logspace(abs(start), 1e-3, res * 9 // 10), np.logspace(1e-3, end, res * 1 // 10))
            else:
                return np.linspace(start, end, res)

        if re is not None:
            self.re_range = get_range(re[0], re[1], res, log)
        if im is not None:
            self.im_range = get_range(im[0], im[1], res, log)
        self.run_dahlquist(**kwargs)

    def run_dahlquist(self, **kwargs):
        '''
        Run a Dahlquist problem on region of the complex plane

        Args:
            some args can be passed to influence the run

        Returns:
            pySDC.stats: Stats object created by the run
        '''

        sweeper_params = {
            **self.sweeper_params,
            'initial_guess': 'random',
        }

        # build lambdas
        lambdas = np.array(
            [
                [complex(self.re_range[i], self.im_range[j]) for i in range(len(self.re_range))]
                for j in range(len(self.im_range))
            ]
        ).reshape((len(self.re_range) * len(self.im_range)))

        problem_params = {
            'lambdas': lambdas,
        }

        dt = kwargs.get('dt', 1.0)
        level_params = {
            'dt': dt,
        }

        desc = {'sweeper_params': sweeper_params, 'sweeper_class': self.data['sweeper'], 'level_params': level_params}
        stats, _, _ = run_dahlquist(custom_description=desc, custom_problem_params=problem_params, Tend=dt, **kwargs)
        self.dahlquist_stats = stats
        return stats

    def plot_dahlquist(self, plot_func, **kwargs):
        """
        Plot something for the Dahlquist equation.

        Args:
            plot_func: Should contain a plotting function from the Dahlquist script

        Returns:
            None
        """

        if not kwargs.get('ax', False) or not kwargs.get('fig', False):
            fig, ax = plt.subplots()
            kwargs['ax'] = ax
            kwargs['fig'] = fig

        if self.dahlquist_stats is None:
            self.run_dahlquist(**kwargs)

        return plot_func(self.dahlquist_stats, **kwargs)

    def get_initial_conditions(self, problem, error=False, **kwargs):
        """
        Get the initial conditions or the initial error for a given problem

        Args:
            problem (str): Name of the problem
            error (bool): Wether to get the error or the intial conditions

        Returns:
            dtype_u: Initial conditions or intial error of the problem
        """
        params = get_params(problem)
        stats, controller, _ = params['prob'](custom_problem_params=params['problem_params'])

        u0 = get_sorted(stats, type='u0')[0][1]
        if not error:
            return u0
        else:
            dt = get_sorted(stats, type='dt')[0][1]
            u_exact = controller.MS[0].levels[0].prob.u_exact(t=dt)
            return u0 - u_exact

    def plot_fourier_decomposition(self, problem, **kwargs):
        """
        Plot the damping of various Fourier modes in the error in space per iteration.

        Args:
            problem (str): Name of a problem to run

        Returns:
            None
        """
        from pySDC.helpers.stats_helper import filter_stats

        # prepare a figure
        if 'ax' not in kwargs.keys():
            fig, ax = plt.subplots()
        else:
            ax = kwargs['ax']

        # assemble params
        replace_params = {
            'convergence_controllers': {},
            'problem_params': get_params(problem)['problem_params'],
        }

        # run the problem
        stats, controller = self.run_problem(
            logger_level=30,
            replace_params=replace_params,
            hook_class=log_error_at_iterations,
            **kwargs,
        )

        steps = kwargs.get('steps', [-1])
        all_u_exact = get_sorted(stats, type='u_exact', sortby='time')

        # prepare some variables for plotting
        k = np.arange(len(all_u_exact[0][1]))
        ax.set_yscale('log')

        for step in steps:
            # filter the step
            stats_step = filter_stats(stats, time=all_u_exact[step][0])

            # get solutions and exact solution
            u_exact = get_sorted(stats_step, type='u_exact', sortby='iter')
            u = get_sorted(stats_step, type='u', sortby='iter')

            # compute the error
            iterations = [me[0] for me in u]
            e = [u[i][1] - u_exact[0][1] for i in iterations]

            # do FFTs of the error
            e_FFT = [np.fft.fft(e[i]) for i in iterations]

            # plot the error for each iteration
            [ax.scatter(k, abs(e_FFT[i]), label=i) for i in iterations]

        if not kwargs.get("no_legend", False):
            ax.legend(frameon=False)
        if not kwargs.get("no_title", False):
            ax.set_title(f'{self.label}')
        if kwargs.get("labels", True):
            ax.set_xlabel(r'$k$')
            ax.set_ylabel('Fourier coefficients')

    def plot_eigenvalues(
        self, problem=None, problem_parameter=1.0, active_only=True, rescale=True, ax=None, fig=None, vmin=-16, **kwargs
    ):
        """
        Plot the eigenvalues of the finite difference discretization corresponding to active frequencies in space in the
        complex plane.

        Args:
            problem (str): Name of the problem to run
            problem_parameter (float): Problem parameter to rescale the finite difference matrix
            active_only (bool): Whether to show only active modes in the problem or all possible ones
            rescale (bool): Show only the region in the complex plane around the eigenvalues of interest
            ax: Somewhere to plot
            fig: Figure containing the ax
            vmin (float): Minimal value for the heatmap

        Returns:
            Contraction factor plot
        """
        # get problem parameters
        params = get_params(problem)
        problem_params = params['problem_params']

        # get the eigenvalues of the Toeplitz matrix
        eigenvals = problem_parameter * get_finite_difference_eigenvalues(
            params['derivative'],
            order=problem_params['order'],
            type=problem_params['type'],
            dx=params['L'] / problem_params['nvars'],
            L=params['L'],
        )

        if active_only:
            # get initial error
            e0 = self.get_initial_conditions(problem, error=True, **kwargs)

            # get the active modes in by checking Fourier coefficients
            fft = np.fft.fft(e0)
            active_modes = abs(fft) > 1e-4 * sum(abs(fft))
        else:
            active_modes = np.ones_like(eigenvals, dtype=bool)

        # change the range in the complex plane
        if rescale:
            real = [min(eigenvals[active_modes].real) * 1.1, max(eigenvals[active_modes].real) * 1.1]
            imag = [min(eigenvals[active_modes].imag) * 1.1, max(eigenvals[active_modes].imag) * 1.1]

            real[0] = min([-1, real[0]])
            real[1] = max([1, real[1]])
            imag[0] = min([-1, imag[0]])
            imag[1] = max([1, imag[1]])
            self.change_dahlquist_range(re=real, im=imag, log=False, **kwargs)

        # plot the contraction factor
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        im = self.plot_dahlquist(plot_contraction, ax=ax, fig=fig, vmin=vmin, **kwargs)

        # plot the eigenvalues
        ax.scatter(eigenvals[active_modes].real, eigenvals[active_modes].imag, color='black', marker='*')

        return im

    def plot_everything(self, **kwargs):
        """
        Plot both stability as well as contraction rate
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        self.plot_dahlquist(plot_stability, ax=axs[0], **kwargs)
        self.plot_dahlquist(plot_contraction, fig=fig, ax=axs[1], **kwargs)
        axs[0].set_xscale('symlog')
        fig.tight_layout()
        return fig

    def run_problem(self, logger_level=15, custom_params=None, replace_params=None, **kwargs):
        """
        Run the problem that the preconditioner was derived with

        Args:
            logger_level (int): Logger level to display more information about the run
            custom_params (dict): Change parameters from the original run
            replace_params (dict): Replace parameters from the original run

        Returns:
            dict: The stats generated by a pySDC run
            controller: The controller used in the run
        """
        params = self.data['params']
        params['controller_params'] = {'logger_level': logger_level}
        if custom_params:
            params = {**params, **custom_params}
        if replace_params:
            for key in replace_params.keys():
                params[key] = replace_params[key]

        stats, controller = single_run(self.data['x'], params, **{**self.data['kwargs'], **kwargs})
        return stats, controller

    def get_stiffness(self, problem, parameter=None, parameter_range=None, ax=None, maxiter=10000, **kwargs):
        """
        Run a problem with a range of parameters from non-stiff to stiff

        Args:
            problem (str): Name of a problem to run
            parameter (str): The name of the parameter to vary
            parameter_range (list): List of values for the parameter
            ax: Somewhere to plot

        Returns:
            list: The parameter range
            list: The total iterations needed to solve the problem
        """
        custom_params, parameter_paper, parameter_range_paper = get_params_for_stiffness_plot(
            problem, **{**self.data['kwargs'], **kwargs}
        )
        custom_problem_params = custom_params['problem_params']

        parameter_range = parameter_range_paper if parameter_range is None else parameter_range
        parameter = parameter_paper if parameter is None else parameter

        iterations = np.zeros_like(parameter_range)

        print(f"Running {problem} with {self.label} preconditioner...")
        for i in range(len(parameter_range)):
            custom_problem_params[parameter] = parameter_range[i]

            custom_params['problem_params'] = {**kwargs.get('custom_problem_params', {}), **custom_problem_params}

            stats, controller = self.run_problem(logger_level=30, custom_params=custom_params, **kwargs)

            iterations[i] = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])

            print(f"\t... with parameter {parameter}={parameter_range[i]:.2e} took {iterations[i]:6.0f} iterations")
            if iterations[i] > maxiter:
                print("\tMax. iterations exceeded. Skipping remaining range...")
                break

        if ax:
            mask = iterations > 0
            ax.plot(
                parameter_range[mask],
                iterations[mask],
                **kwargs.get('plotting_params', {}),
                color=self.color,
                ls=self.ls,
            )
        return parameter, parameter_range, iterations


def compare_stiffness(precons, problem, parameter=None, parameter_range=None, **kwargs):
    """
    Run a problem with a range of parameters from non-stiff to stiff for a range of preconditioners

    Args:
        precons (list): List of preconditioners as objects of the PreconPostProcessing class
        problem (str): Name of a problem to run
        parameter (str): The name of the parameter to vary
        parameter_range (list): List of values for the parameter
        ax: Somewhere to plot

    Returns:
        list: The parameter range
        list: The total iterations needed to solve the problem
    """
    if 'ax' not in kwargs.keys():
        fig, ax = plt.subplots(1, 1)
        kwargs['ax'] = ax
    else:
        ax = kwargs['ax']

    for i in range(len(precons)):
        plotting_params = {'label': precons[i].label}
        parameter, _, _ = precons[i].get_stiffness(
            problem, parameter, parameter_range, plotting_params=plotting_params, **kwargs
        )

    if kwargs.get('legend', True):
        ax.legend(frameon=False)
    ax.set_xlabel(parameter)
    ax.set_ylabel('number of iterations')

    if kwargs.get("logx", False):
        ax.set_xscale('log')
    if kwargs.get("logy", False):
        ax.set_yscale('log')

    if 'ax' not in kwargs.keys():
        fig.tight_layout()
        plt.savefig(f'data/plots/stiffness-{problem}-{parameter}.{kwargs.get("format", "pdf")}', bbox_inches='tight')


def compare_stiffness_paper(precons, **kwargs):
    """
    Reproduce the plots in Robert's paper "Parallelizing spectral deferred corrections across the method", where he
    plots the number of iterations needed for a problem with a given preconditioner in a range from non-stiff to stiff
    configurations.

    Args:
        precons (list): List of preconditioners as objects of the PreconPostProcessing class

    Returns:
        None
    """
    problems = ['heat', 'advection', 'vdp']
    xlabels = [r'$\nu$', r'$c$', r'$\mu$']

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))

    for i in range(len(problems)):
        compare_stiffness(precons, problem=problems[i], ax=axs[i], logx=True, adaptivity=False, legend=i == 0)
        if i > 0:
            axs[i].set_ylabel(None)
        axs[i].set_title(problems[i])
        axs[i].set_xlabel(xlabels[i])
    fig.tight_layout()
    plt.savefig(f'data/plots/stiffness.{kwargs.get("format", "pdf")}', bbox_inches='tight', dpi=200)


def compare_Fourier(precons, problem, **kwargs):
    """
    Compare the damping of Fourier modes in space for various preconditioners

    Args:
        precons (list): List of preconditioners as PreconPostProcessing objects
        problem (str): Name of a problem to run

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for i in range(len(precons)):
        precons[i].plot_fourier_decomposition(problem=problem, ax=axs.flat[i], no_legend=i > 0, labels=i == 2)
    fig.tight_layout()
    plt.savefig(f'data/plots/fourier_comparison-{problem}.{kwargs.get("format", "pdf")}', dpi=200, bbox_inches='tight')


def compare_contraction(precons, plot_eigenvals=False, log=False, vmin=1e-16, **kwargs):
    """
    Make some plots of contraction factors accross multiple preconditioners

    Args:
        precons (list): List of preconditioners as PreconPostProcessing objects
        plot_eigenvals (bool): Whether to plot the eigenvalues of the finite difference discretization. Requires
                               additional keyword arguments
        log (bool): Wheter to put a symlog scaling on the axes
        vmin (float): Minimum for the heatmaps

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True)
    axs = axs.flatten()

    real = [-7, 0]
    imag = [-8, 8]

    for i in range(len(axs)):
        if plot_eigenvals:
            im = precons[i].plot_eigenvalues(ax=axs[i], fig=fig, cbar=False, **kwargs)
        else:
            precons[i].change_dahlquist_range(re=real, im=imag, res=100, log=True)
            im = precons[i].plot_dahlquist(plot_contraction, fig=fig, ax=axs[i], iter=[0, 4], vmin=vmin, cbar=False)
        axs[i].set_title(precons[i].label)
        if log:
            axs[i].set_xscale('symlog')
            axs[i].set_yscale('symlog')

    cb = fig.colorbar(im, ax=axs.ravel().tolist())
    cb.set_label(r'$\log \rho$')

    plt.savefig(
        f'data/plots/contraction_comparison{"-" if plot_eigenvals else ""}{kwargs.get("problem", "")}.{kwargs.get("format", "pdf")}',
        dpi=200,
        bbox_inches='tight',
    )


def generate_metadata_table(precons, path='./data/notes/metadata.md'):
    # TODO: docs...
    with open(path, 'w') as file:
        # print some title etc
        file.write('# Preconditioners \nWe supply some information about the preconditioners here\n')

        # print header
        file.write('| name | source | parallelizable | normalized | semi-diagonal | random IG |\n')
        file.write('|------|--------|----------------|------------|---------------|-----------|\n')

        # print data for the precons
        for precon in precons:
            file.write(
                f'| {precon.label} | {precon.source} | {precon.parallelizable} | {precon.normalized} | {precon.semi_diagonal} | {precon.random_ig} |\n'
            )


kwargs = {
    'adaptivity': True,
    'random_IG': True,
}

problem = 'heat'
problem_serial = 'advection'

postLU = PreconPostProcessing(
    problem_serial,
    3,
    LU=True,
    label='LU',
    color=colors[0],
    ls='--',
    source='[Martin Weiser](https://doi.org/10.1007/s10543-014-0540-y)',
    **kwargs,
)
postIE = PreconPostProcessing(
    problem_serial,
    3,
    IE=True,
    label='Implicit Euler',
    color=colors[1],
    ls='--',
    source='[Dutt et al.](https://doi.org/10.1023/A:1022338906936)',
    **kwargs,
)
postDiag = PreconPostProcessing(problem, 3, label='Diagonal', color=colors[2], **kwargs)
postMIN3 = PreconPostProcessing(
    problem, 3, MIN3=True, label='MIN3', color=colors[6], ls='-.', parallelizable=True, source='Anonymous', **kwargs
)
postDiagFirstRow = PreconPostProcessing(
    problem,
    3,
    **kwargs,
    use_first_row=True,
    color=colors[3],
    label='Semi-Diagonal',
    parallelizable=True,
)
postMIN = PreconPostProcessing(
    problem_serial,
    3,
    MIN=True,
    label='MIN',
    color=colors[4],
    ls='-.',
    source='[Robert](https://doi.org/10.1007/s00791-018-0298-x)',
    parallelizable=True,
    **kwargs,
)
postNORM = PreconPostProcessing(
    problem, 3, **kwargs, normalized=True, label='normalized', color=colors[5], ls='-', parallelizable=True
)

precons = [postDiagFirstRow, postLU, postDiag, postIE]
more_precons = precons + [postMIN, postNORM, postMIN3]

custom_problem_params = {
    'sigma': 6e-2,
    'freq': -1,
    'order': 2,
}

pkwargs = {'Tend': 1e-2}


# compare_contraction(precons, plot_eigenvals= True, problem='advection', problem_parameter=-1, vmin=-9)
# compare_contraction(precons, plot_eigenvals=True, problem_parameter=1, vmin=-10, problem='heat')
# compare_Fourier(precons, problem='heat')
# compare_Fourier(precons, problem='advection')
generate_metadata_table(more_precons)
compare_stiffness_paper(more_precons, format='png')
fig, axs = plt.subplots(1, 2, figsize=(11.1, 4.1))
active_only = True
precon = postIE
im = precon.plot_eigenvalues(
    problem='heat',
    problem_parameter=1.0,
    active_only=active_only,
    rescale=True,
    ax=axs[0],
    vmin=-16,
    vmax=0.0,
    fig=fig,
    cbar=False,
    iter=[0, 4],
    dt=5.0 / 60.0,
)
precon.plot_eigenvalues(
    problem='advection',
    problem_parameter=-1.0,
    active_only=active_only,
    rescale=True,
    ax=axs[1],
    vmin=-16,
    vmax=0.0,
    fig=fig,
    cbar=False,
    iter=[0, 4],
    res=500,
    dt=5.0 / 60.0,
)
axs[0].set_title('heat')
axs[1].set_title('advection')
axs[0].set_xlabel(r'Re($\lambda \Delta t$)')
axs[0].set_ylabel(r'Im($\lambda \Delta t$)')
cb = fig.colorbar(im, ax=axs.ravel().tolist())
cb.set_label(r'$\log \rho$')
# if not active_only:
#    plt.savefig('data/notes/rho-IE-FD-eigenvals.png', dpi=200, bbox_inches='tight')
# else:
#    plt.savefig('data/notes/rho-IE-FD-eigenvals-active-dt.png', dpi=200, bbox_inches='tight')
plt.show()
