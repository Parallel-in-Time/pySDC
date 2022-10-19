import matplotlib.pyplot as plt
import numpy as np

from pySDC.projects.Resilience.dahlquist import run_dahlquist, plot_stability, plot_contraction
from pySDC.playgrounds.Preconditioners.optimize_diagonal_preconditioner import single_run
from pySDC.playgrounds.Preconditioners.configs import load_precon


class PreconPostProcessing:
    def __init__(self, problem, nodes, **kwargs):
        '''
        Load a preconditioner for postprocessing

        Args:
            problem (str): The name of the problem that has been run to obtain the preconditioner
            nodes (int): Number of nodes that have been used
        '''
        # laod data
        self.data = load_precon(problem, nodes, **kwargs)
        self.diagonal_elements = self.data['diags']
        self.first_row = self.data['first_row']

        # prepare empty variables
        self.dahlquist_stats = None
        self.re_range = np.linspace(-30, 30, 400)
        self.im_range = np.linspace(-50, 50, 400)

    def change_dahlquist_range(self, re=None, im=None, res=400):
        '''
        Change the range in the complex plane for running the Dahlquist problems and rerun them

        Args:
            re (tuple): Range for the real part
            im (tuple): Range for the imaginary part
            res (int): Number of entries per direction

        Returns:
            None
        '''
        if re is not None:
            self.re_range = np.linspace(re[0], re[1], res)
        if im is not None:
            self.im_range = np.linspace(im[0], im[1], res)
        self.run_dahlquist()

    def run_dahlquist(self, **kwargs):
        '''
        Run a Dahlquist problem on region of the complex plane

        Args:
            some args can be passed to influence the run

        Returns:
            pySDC.stats: Stats object created by the run
        '''

        sweeper_params = {
            'diagonal_elements': self.diagonal_elements,
            'first_row': self.first_row,
            'num_nodes': self.data['num_nodes'],
            'quad_type': self.data['quad_type'],
            'QI': self.data['QI']
        }

        # build lambdas
        lambdas = np.array([[complex(self.re_range[i], self.im_range[j]) for i in range(len(self.re_range))]
                            for j in range(len(self.im_range))]).reshape((len(self.re_range) * len(self.im_range)))

        problem_params = {
            'lambdas': lambdas,
        }

        desc = {
            'sweeper_params': sweeper_params,
            'sweeper_class': self.data['params']['sweeper']
        }
        stats, _, _ = run_dahlquist(custom_description=desc, custom_problem_params=problem_params, **kwargs)
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
            self.run_dahlquist()

        plot_func(self.dahlquist_stats, **kwargs)

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

    def run_problem(self, logger_level=15, **kwargs):
        """
        Run the problem that the preconditioner was derived with

        Args:
            logger_level (int): Logger level to display more information about the run
        """
        params = self.data['params']
        params['controller_params'] = {'logger_level': logger_level}
        stats, controller = single_run(self.data['x'], params, **self.data['kwargs'])


kwargs = {
    'adaptivity': True
}

postLU = PreconPostProcessing('advection', 3, LU=True, **kwargs)
postIE = PreconPostProcessing('advection', 3, IE=True, **kwargs)
postDiag = PreconPostProcessing('advection', 3, **kwargs)
postDiagFirstRow = PreconPostProcessing('advection', 3, adaptivity=True, use_first_row=True)
posts = [postDiagFirstRow, postLU, postDiag, postIE]

postDiag.run_problem()

fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
axs = axs.flatten()
posts[1].change_dahlquist_range(re=[-1000, 0], im=[-100, 100])
for i in range(len(axs)):
    posts[i].change_dahlquist_range(re=[-1000, 1], im=[-100, 100])
    posts[i].plot_dahlquist(plot_contraction, fig=fig, ax=axs[i], iter=[0, 4])
    axs[i].set_xscale('symlog')
fig.tight_layout()
plt.savefig('data/plots/contraction_comparison.pdf', dpi=200)
plt.show()
