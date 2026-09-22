import logging
from typing import Any, Dict

import numpy as np


class ParaDiag:
    """
    What ParaDiag is, independently of how the block is spread over processes.

    Mixed into a controller, this replaces its iteration and leaves everything around it alone:

        class controller_ParaDiag_nonMPI(ParaDiag, controller_nonMPI)
        class controller_ParaDiag_MPI(ParaDiag, controller_MPI)

    It deliberately has no `__init__` and no controller of its own to inherit from, so that it can
    be mixed into either transport's controller without the two having to agree on a constructor
    signature. A concrete ParaDiag controller calls `prepare_ParaDiag_params` on the two
    dictionaries and then hands them to its controller's initialisation.

    What stays with the concrete classes is everything whose implementation depends on where the
    other steps are: `apply_matrix`, `prepare_Jacobians`, `compute_all_at_once_residual`,
    `update_G_inv` and the block driver.
    """

    @staticmethod
    def prepare_ParaDiag_params(controller_params: Dict[str, Any], description: Dict[str, Any]) -> None:
        """
        Check and complete the parameters ParaDiag needs, in place.

        Call this *before* the controller's own initialisation: it only reads and writes the two
        dictionaries, and must have run by the time the steps are built.

        Args:
            controller_params (dict): parameter set for the controller and the steps
            description (dict): all the parameters to set up the rest (levels, problems, ...)
        """
        from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

        if QDiagonalization in description['sweeper_class'].__mro__:
            description['sweeper_params']['ignore_ic'] = True
            description['sweeper_params']['update_f_evals'] = False
        else:
            logging.getLogger('controller').warning(
                f'Warning: Your sweeper class {description["sweeper_class"]} is not derived from {QDiagonalization}. You probably want to use another sweeper class.'
            )

        if not controller_params.get('all_to_done', True):
            raise NotImplementedError('ParaDiag only implemented with option `all_to_done=True`')
        if 'alpha' not in controller_params.keys():
            from pySDC.core.errors import ParameterError

            raise ParameterError('Please supply alpha as a parameter to the ParaDiag controller!')
        controller_params['average_jacobian'] = controller_params.get('average_jacobian', True)

        controller_params['all_to_done'] = True

    # ------------------------------------------------------------------ the iteration

    def get_stages(self) -> Dict[str, Any]:
        """
        ParaDiag has one iteration stage, and no predictor because it has no coarse level.

        Returns:
            dict: stage name -> the method that runs it
        """
        return {
            'SPREAD': self.spread,
            'IT_CHECK': self.it_check,
            'IT_PARADIAG': self.it_ParaDiag,
        }

    def next_iteration_stage(self, S: Any) -> str:
        """
        Args:
            S (pySDC.Step.step): The current step

        Returns:
            str: name of the stage to enter
        """
        return 'IT_PARADIAG'

    def compute_residual_after_spread(self, S: Any) -> None:
        """
        ParaDiag's residual is the one of the composite collocation problem, which `it_ParaDiag`
        computes as part of the iteration. The convergence check runs before the first iteration,
        so the initial guess needs its residual here.

        Args:
            S (pySDC.Step.step): The current step
        """
        S.levels[0].sweep.compute_residual()

    def step_is_active(self, time: float, block_start: float, Tend: float) -> bool:
        """
        ParaDiag diagonalizes across the whole block, so it cannot drop a step out of one. A block
        that starts before `Tend` is run whole, past `Tend` if need be.

        Args:
            time (float): when this step starts
            block_start (float): when the first step of this step's block starts
            Tend (float): ending time

        Returns:
            bool: whether this step takes part
        """
        active = block_start < Tend - 10 * np.finfo(float).eps

        if active and time >= Tend - 10 * np.finfo(float).eps:
            self.logger.warning(
                'Warning: This controller will solve past your desired end time until the end of its block!'
            )

        return active

    def prepare_convergence_check(self, *args: Any, **kwargs: Any) -> None:
        """
        The residual is already current -- `it_ParaDiag` computed it, and recomputing it the way a
        sweep-based algorithm does would need initial conditions that have not been communicated
        yet. The end point is not, because nothing sent it anywhere, so compute it here: `it_check`
        is what publishes `uend` when a step is done.

        Takes whatever its controller passes -- a block of steps or a communicator -- and needs
        none of it, because the steps to do this for are the ones this controller owns.
        """
        for S in self.steps:
            S.levels[0].sweep.compute_end_point()

    # ------------------------------------------------------------------ alpha and the transform

    @staticmethod
    def resolve_alpha(alpha: Any, k: int = 0) -> float:
        """
        Read the alpha for iteration `k` out of whatever the user supplied.

        `alpha` may be a single number, a sequence indexed by iteration (the last entry is reused once
        it runs out), or a callable taking the iteration index. Making it iteration dependent lets the
        outer iteration start with a well-conditioned alpha and tighten it later.

        Static because the steps need an alpha before the controller has parameters to read it from.

        Args:
            alpha: the alpha parameter as supplied by the user
            k (int): iteration index

        Returns:
            float: alpha to use for this iteration
        """
        if callable(alpha):
            return float(alpha(k))
        if hasattr(alpha, '__len__'):
            return float(alpha[min(k, len(alpha) - 1)])
        return float(alpha)

    def get_alpha(self, k: int = 0) -> float:
        """
        Get the ParaDiag alpha parameter for iteration `k`.

        Args:
            k (int): iteration index

        Returns:
            float: alpha to use for this iteration
        """
        return self.resolve_alpha(self.params.alpha, k)

    def get_FFT_matrices(self, k: int = 0) -> Any:
        """
        Get the weighted FFT and iFFT matrices for iteration `k`, rebuilding them only when alpha
        actually changes.

        Args:
            k (int): iteration index

        Returns:
            tuple: the forward and backward weighted FFT matrices
        """
        alpha = self.get_alpha(k)
        if getattr(self, '_cached_alpha', None) != alpha:
            from pySDC.helpers.ParaDiagHelper import get_weighted_FFT_matrix, get_weighted_iFFT_matrix

            self._FFT_matrix = get_weighted_FFT_matrix(self.n_steps, alpha)
            self._iFFT_matrix = get_weighted_iFFT_matrix(self.n_steps, alpha)
            self._cached_alpha = alpha
        return self._FFT_matrix, self._iFFT_matrix

    def FFT_in_time(self, quantity: Any, k: int = 0) -> None:
        """
        Compute weighted forward FFT in time. The weighting is determined by the alpha parameter in ParaDiag

        Note: The implementation via matrix-vector multiplication may be inefficient and less stable compared to an FFT
              with transposes!

        Args:
            quantity (str): the level attribute to transform
            k (int): iteration index, for an iteration dependent alpha
        """
        self.apply_matrix(self.get_FFT_matrices(k)[0], quantity)

    def iFFT_in_time(self, quantity: Any, k: int = 0) -> None:
        """
        Compute weighted backward FFT in time. The weighting is determined by the alpha parameter in ParaDiag

        Args:
            quantity (str): the level attribute to transform
            k (int): iteration index, for an iteration dependent alpha
        """
        self.apply_matrix(self.get_FFT_matrices(k)[1], quantity)

    # ------------------------------------------------------------------ what the transport supplies

    def apply_matrix(self, mat: Any, quantity: str) -> None:
        """
        Apply a square matrix across the steps, in place.

        How this is done depends entirely on where the other steps are, so the concrete controllers
        implement it.

        Args:
            mat: square matrix with as many rows as there are steps
            quantity (str): 'residual' or 'increment', the level attribute to transform
        """
        raise NotImplementedError('ParaDiag controllers have to implement apply_matrix')

    def update_G_inv(self, k: int = 0) -> None:
        """
        Rebuild G^-1 on the local step(s) when alpha changes with the iteration.

        G^-1 depends on alpha, so an iteration dependent alpha means the sweeper's diagonalization has
        to be recomputed. Subclasses implement this because only they know which steps they own.

        Args:
            k (int): iteration index
        """
        raise NotImplementedError('ParaDiag controllers have to implement update_G_inv')
