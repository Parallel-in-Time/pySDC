import numpy as np

from pySDC.core.convergence_controller import ConvergenceController


class AdaptiveAlpha(ConvergenceController):
    r"""
    Choose the ParaDiag :math:`\alpha` adaptively from the residual.

    ParaDiag replaces the time-stepping matrix by an :math:`\alpha`-circulant approximation, and
    :math:`\alpha` trades two error sources against each other: a small value approximates the original
    problem better, but conditions the diagonalization worse, so round-off and inexact inner solves
    contaminate the result. A single fixed :math:`\alpha` therefore has to be a compromise for the whole
    run, even though the balance shifts as the residual falls.

    This convergence controller updates :math:`\alpha` after every iteration instead, following the
    strategy in `Čaklović et al. <https://doi.org/10.2140/camcos.2023.18.55>`_:

    .. math::
        \gamma = L (3 \epsilon + \tau), \quad
        \alpha_{k} = \sqrt{\frac{\gamma r_k}{e_k}}, \quad
        e_{k+1} = 2 \sqrt{\gamma e_k r_k},

    with :math:`L` the number of steps in the block, :math:`\epsilon` machine precision, :math:`\tau`
    the inner solver tolerance, :math:`r_k` the residual and :math:`e_k` a running bound on the error.
    :math:`\gamma` is the accuracy floor: there is no point pushing :math:`\alpha` below the level at
    which round-off and the inner solver dominate anyway.

    The residual is reduced over the whole block, so every rank computes the same :math:`\alpha` and the
    controllers stay in step.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            'control_order': +300,
            # accuracy floor: round-off plus whatever the inner solver leaves behind
            'inner_tol': 0.0,
            # initial bound on the error, before we have seen a residual
            'e0': 1.0,
            # keep alpha in a sane range no matter what the residual does
            'alpha_min': 1e-12,
            'alpha_max': 1.0,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def setup_status_variables(self, controller, **kwargs):
        """
        Start the alpha history, which spans the whole run rather than a single block.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.alphas = []
        return None

    def reset_status_variables(self, controller, **kwargs):
        """
        Reset the error bound at the start of every block.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.e = self.params.e0
        self.last_iter = None
        return None

    def get_gamma(self, controller):
        r"""
        The accuracy floor :math:`\gamma = L (3 \epsilon + \tau)`.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            float: gamma
        """
        eps = np.finfo(complex).eps
        return controller.n_steps * (3 * eps + self.params.inner_tol)

    def post_iteration_processing(self, controller, S, **kwargs):
        r"""
        Compute the next :math:`\alpha` from the residual of the whole block.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # The virtually parallel controller calls this once per step, the MPI one once per rank. Either
        # way alpha must advance once per iteration, or the recursion below runs L times too fast.
        if self.last_iter == S.status.iter:
            return None
        self.last_iter = S.status.iter

        # the residual of the composite problem is the largest one across the block
        residual = max(step.levels[0].status.residual for step in controller.steps)

        comm = kwargs.get('comm', None)
        if comm is not None:
            residual = comm.allreduce(residual, op=self.MPI_MAX)

        if residual <= 0:
            return None

        gamma = self.get_gamma(controller)
        alpha = np.sqrt(gamma * residual / self.e)
        self.e = 2 * np.sqrt(gamma * self.e * residual)

        controller.params.alpha = min(max(alpha, self.params.alpha_min), self.params.alpha_max)
        self.debug(f'Set alpha to {controller.params.alpha:.3e} from residual {residual:.3e}', S)

        # keep the history around; it is what you want to look at when tuning
        self.alphas.append(controller.params.alpha)

        return None

    def dependencies(self, controller, description, **kwargs):
        """
        Prepare the MPI reduction we need for the block residual.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        if self.params.useMPI:
            self.prepare_MPI_datatypes()
            from mpi4py import MPI

            self.MPI_MAX = MPI.MAX

        super().dependencies(controller, description, **kwargs)
        return None
