from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.spectral_helper import SpectralHelper
import numpy as np
from pySDC.core.errors import ParameterError


class GenericSpectralLinear(Problem):
    """
    Generic class to solve problems of the form M u_t + L u = y, with mass matrix M, linear operator L and some right
    hand side y using spectral methods.
    L may contain algebraic conditions, as long as (M + dt L) is invertible.

    Note that the `__getattr__` method is overloaded to pass requests on to the spectral helper if they are not
    attributes of this class itself. For instance, you can add a BC by calling `self.spectral.add_BC` or equivalently
    `self.add_BC`.

    You can port problems derived from this more or less seamlessly to GPU by using the numerical libraries that are
    class attributes of the spectral helper. This class will automatically switch the datatype using the `setup_GPU` class method.

    Attributes:
        spectral (pySDC.helpers.spectral_helper.SpectralHelper): Spectral helper
        work_counters (dict): Dictionary for counting work
        cached_factorizations (dict): Dictionary of cached matrix factorizations for solving
        L (sparse matrix): Linear operator
        M (sparse matrix): Mass matrix
        diff_mask (list): Mask for separating differential and algebraic terms
        Pl (sparse matrix): Left preconditioner
        Pr (sparse matrix): Right preconditioner
    """

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh
        from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

        cls.dtype_u = cupy_mesh

        GPU_versions = {
            mesh: cupy_mesh,
            imex_mesh: imex_cupy_mesh,
        }

        cls.dtype_f = GPU_versions[cls.dtype_f]

    def __init__(
        self,
        bases,
        components,
        comm=None,
        Dirichlet_recombination=True,
        left_preconditioner=True,
        solver_type='cached_direct',
        solver_args=None,
        useGPU=False,
        max_cached_factorizations=12,
        spectral_space=True,
        real_spectral_coefficients=False,
        debug=False,
    ):
        """
        Base class for problems discretized with spectral methods.

        Args:
            bases (list of dictionaries): 1D Bases
            components (list of strings): Components of the equations
            comm (mpi4py.Intracomm or None): MPI communicator
            Dirichlet_recombination (bool): Use Dirichlet recombination in the last axis as right preconditioner
            left_preconditioner (bool): Reverse the Kronecker product if yes
            solver_type (str): Solver for linear systems
            solver_args (dict): Arguments for linear solver
            useGPU (bool): Run on GPU or CPU
            max_cached_factorizations (int): Number of matrix decompositions to cache before starting eviction
            spectral_space (bool): If yes, the solution will not be transformed back after solving and evaluating the RHS, and is expected as input in spectral space to these functions
            real_spectral_coefficients (bool): If yes, allow only real values in spectral space, otherwise, allow complex.
            debug (bool): Make additional tests at extra computational cost
        """
        solver_args = {} if solver_args is None else solver_args
        self._makeAttributeAndRegister(
            'max_cached_factorizations',
            'useGPU',
            'solver_type',
            'solver_args',
            'left_preconditioner',
            'Dirichlet_recombination',
            'comm',
            'spectral_space',
            'real_spectral_coefficients',
            'debug',
            localVars=locals(),
        )
        self.spectral = SpectralHelper(comm=comm, useGPU=useGPU, debug=debug)

        if useGPU:
            self.setup_GPU()

        for base in bases:
            self.spectral.add_axis(**base)
        self.spectral.add_component(components)

        self.spectral.setup_fft(real_spectral_coefficients)

        super().__init__(init=self.spectral.init_forward if spectral_space else self.spectral.init)

        self.work_counters[solver_type] = WorkCounter()
        self.work_counters['factorizations'] = WorkCounter()

        self.setup_preconditioner(Dirichlet_recombination, left_preconditioner)

        self.cached_factorizations = {}

    def __getattr__(self, name):
        """
        Pass requests on to the helper if they are not directly attributes of this class for convenience.

        Args:
            name (str): Name of the attribute you want

        Returns:
            request
        """
        return getattr(self.spectral, name)

    def _setup_operator(self, LHS):
        """
        Setup a sparse linear operator by adding relationships. See documentation for ``GenericSpectralLinear.setup_L`` to learn more.

        Args:
            LHS (dict): Equations to be added to the operator

        Returns:
            sparse linear operator
        """
        operator = self.spectral.get_empty_operator_matrix()
        for line, equation in LHS.items():
            self.spectral.add_equation_lhs(operator, line, equation)
        return self.spectral.convert_operator_matrix_to_operator(operator)

    def setup_L(self, LHS):
        """
        Setup the left hand side of the linear operator L and store it in ``self.L``.

        The argument is meant to be a dictionary with the line you want to write the equation in as the key and the relationship between components as another dictionary. For instance, you can add an algebraic condition capturing a first derivative relationship between u and ux as follows:

        ```
        Dx = self.get_differentiation_matrix(axes=(0,))
        I = self.get_Id()
        LHS = {'ux': {'u': Dx, 'ux': -I}}
        self.setup_L(LHS)
        ```

        If you put zero as right hand side for the solver in the line for ux, ux will contain the x-derivative of u afterwards.

        Args:
            LHS (dict): Dictionary containing the equations.
        """
        self.L = self._setup_operator(LHS)

    def setup_M(self, LHS):
        '''
        Setup mass matrix, see documentation of ``GenericSpectralLinear.setup_L``.
        '''
        diff_index = list(LHS.keys())
        self.diff_mask = [me in diff_index for me in self.components]
        self.M = self._setup_operator(LHS)

    def setup_preconditioner(self, Dirichlet_recombination=True, left_preconditioner=True):
        """
        Get left and right preconditioners.

        Args:
            Dirichlet_recombination (bool): Basis conversion for right preconditioner. Useful for Chebychev and Ultraspherical methods. 10/10 would recommend.
            left_preconditioner (bool): If True, it will interleave the variables and reverse the Kronecker product
        """
        sp = self.spectral.sparse_lib
        N = np.prod(self.init[0][1:])

        Id = sp.eye(N)
        Pl_lhs = {comp: {comp: Id} for comp in self.components}
        self.Pl = self._setup_operator(Pl_lhs)

        if left_preconditioner:
            # reverse Kronecker product

            if self.spectral.useGPU:
                R = self.Pl.get().tolil() * 0
            else:
                R = self.Pl.tolil() * 0

            for j in range(self.ncomponents):
                for i in range(N):
                    R[i * self.ncomponents + j, j * N + i] = 1.0

            self.Pl = self.spectral.sparse_lib.csc_matrix(R)

        if Dirichlet_recombination and type(self.axes[-1]).__name__ in ['ChebychevHelper, Ultraspherical']:
            _Pr = self.spectral.get_Dirichlet_recombination_matrix(axis=-1)
        else:
            _Pr = Id

        Pr_lhs = {comp: {comp: _Pr} for comp in self.components}
        self.Pr = self._setup_operator(Pr_lhs) @ self.Pl.T

    def solve_system(self, rhs, dt, u0=None, *args, skip_itransform=False, **kwargs):
        """
        Do an implicit Euler step to solve M u_t + Lu = rhs, with M the mass matrix and L the linear operator as setup by
        ``GenericSpectralLinear.setup_L`` and ``GenericSpectralLinear.setup_M``.

        The implicit Euler step is (M - dt L) u = M rhs. Note that M need not be invertible as long as (M + dt*L) is.
        This means solving with dt=0 to mimic explicit methods does not work for all problems, in particular simple DAEs.

        Note that by putting M rhs on the right hand side, this function can only solve algebraic conditions equal to
        zero. If you want something else, it should be easy to overload this function.
        """

        sp = self.spectral.sparse_lib

        if self.spectral_space:
            rhs_hat = rhs.copy()
        else:
            rhs_hat = self.spectral.transform(rhs)

        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(rhs_hat.shape)
        rhs_hat = self.spectral.put_BCs_in_rhs_hat(rhs_hat)
        rhs_hat = self.Pl @ rhs_hat.flatten()

        if dt not in self.cached_factorizations.keys():
            A = self.M + dt * self.L
            A = self.Pl @ self.spectral.put_BCs_in_matrix(A) @ self.Pr

        # import numpy as np
        # if A.shape[0] < 200:
        #     import matplotlib.pyplot as plt

        #     # M = self.spectral.put_BCs_in_matrix(self.L.copy())
        #     M = A  # self.L
        #     im = plt.imshow((M / abs(M)).real)
        #     # im = plt.imshow(np.log10(abs(A.toarray())).real)
        #     # im = plt.imshow(((A.toarray())).real)
        #     plt.colorbar(im)
        #     plt.show()

        if self.solver_type.lower() == 'cached_direct':
            if dt not in self.cached_factorizations.keys():
                if len(self.cached_factorizations) >= self.max_cached_factorizations:
                    self.cached_factorizations.pop(list(self.cached_factorizations.keys())[0])
                    self.logger.debug(f'Evicted matrix factorization for {dt=:.6f} from cache')
                self.cached_factorizations[dt] = self.spectral.linalg.factorized(A)
                self.logger.debug(f'Cached matrix factorization for {dt=:.6f}')
                self.work_counters['factorizations']()

            _sol_hat = self.cached_factorizations[dt](rhs_hat)
            self.logger.debug(f'Used cached matrix factorization for {dt=:.6f}')

        elif self.solver_type.lower() == 'direct':
            _sol_hat = sp.linalg.spsolve(A, rhs_hat)
        elif self.solver_type.lower() == 'gmres':
            _sol_hat, _ = sp.linalg.gmres(
                A,
                rhs_hat,
                x0=u0.flatten(),
                **self.solver_args,
                callback=self.work_counters[self.solver_type],
                callback_type='legacy',
            )
        elif self.solver_type.lower() == 'cg':
            _sol_hat, _ = sp.linalg.cg(
                A, rhs_hat, x0=u0.flatten(), **self.solver_args, callback=self.work_counters[self.solver_type]
            )
        else:
            raise NotImplementedError(f'Solver {self.solver_type:!} not implemented in {type(self).__name__}!')

        sol_hat = self.spectral.u_init_forward
        sol_hat[...] = (self.Pr @ _sol_hat).reshape(sol_hat.shape)

        if self.spectral_space:
            return sol_hat
        else:
            sol = self.spectral.u_init
            sol[:] = self.spectral.itransform(sol_hat).real

            if self.spectral.debug:
                self.spectral.check_BCs(sol)

            return sol


def compute_residual_DAE(self, stage=''):
    """
    Computation of the residual that does not add u_0 - u_m in algebraic equations.

    Args:
        stage (str): The current stage of the step the level belongs to
    """

    # get current level and problem description
    L = self.level

    # Check if we want to skip the residual computation to gain performance
    # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
    if stage in self.params.skip_residual_computation:
        L.status.residual = 0.0 if L.status.residual is None else L.status.residual
        return None

    # check if there are new values (e.g. from a sweep)
    # assert L.status.updated

    # compute the residual for each node

    # build QF(u)
    res_norm = []
    res = self.integrate()
    mask = L.prob.diff_mask
    for m in range(self.coll.num_nodes):
        res[m][mask] += L.u[0][mask] - L.u[m + 1][mask]
        # add tau if associated
        if L.tau[m] is not None:
            res[m] += L.tau[m]
        # use abs function from data type here
        res_norm.append(abs(res[m]))
        # print(m, [abs(me) for me in res[m]], [abs(me) for me in L.u[0] - L.u[m + 1]])

    # find maximal residual over the nodes
    if L.params.residual_type == 'full_abs':
        L.status.residual = max(res_norm)
    elif L.params.residual_type == 'last_abs':
        L.status.residual = res_norm[-1]
    elif L.params.residual_type == 'full_rel':
        L.status.residual = max(res_norm) / abs(L.u[0])
    elif L.params.residual_type == 'last_rel':
        L.status.residual = res_norm[-1] / abs(L.u[0])
    else:
        raise ParameterError(
            f'residual_type = {L.params.residual_type} not implemented, choose '
            f'full_abs, last_abs, full_rel or last_rel instead'
        )

    # indicate that the residual has seen the new values
    L.status.updated = False

    return None


def compute_residual_DAE_MPI(self, stage=None):
    """
    Computation of the residual using the collocation matrix Q

    Args:
        stage (str): The current stage of the step the level belongs to
    """
    from mpi4py import MPI

    L = self.level

    # Check if we want to skip the residual computation to gain performance
    # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
    if stage in self.params.skip_residual_computation:
        L.status.residual = 0.0 if L.status.residual is None else L.status.residual
        return None

    # compute the residual for each node

    # build QF(u)
    res = self.integrate(last_only=L.params.residual_type[:4] == 'last')
    mask = L.prob.diff_mask
    res[mask] += L.u[0][mask] - L.u[self.rank + 1][mask]
    # add tau if associated
    if L.tau[self.rank] is not None:
        res += L.tau[self.rank]
    # use abs function from data type here
    res_norm = abs(res)

    # find maximal residual over the nodes
    if L.params.residual_type == 'full_abs':
        L.status.residual = self.comm.allreduce(res_norm, op=MPI.MAX)
    elif L.params.residual_type == 'last_abs':
        L.status.residual = self.comm.bcast(res_norm, root=self.comm.size - 1)
    elif L.params.residual_type == 'full_rel':
        L.status.residual = self.comm.allreduce(res_norm / abs(L.u[0]), op=MPI.MAX)
    elif L.params.residual_type == 'last_rel':
        L.status.residual = self.comm.bcast(res_norm / abs(L.u[0]), root=self.comm.size - 1)
    else:
        raise NotImplementedError(f'residual type \"{L.params.residual_type}\" not implemented!')

    # indicate that the residual has seen the new values
    L.status.updated = False

    return None


def get_extrapolated_error_DAE(self, S, **kwargs):
    """
    The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
    solution obtained by the time marching scheme. This function can be used in `EstimateExtrapolationError`.

    Args:
        S (pySDC.Step): The current step

    Returns:
        None
    """
    u_ex = self.get_extrapolated_solution(S)
    diff_mask = S.levels[0].prob.diff_mask
    if u_ex is not None:
        S.levels[0].status.error_extrapolation_estimate = (
            abs((u_ex - S.levels[0].u[-1])[diff_mask]) * self.coeff.prefactor
        )
        # print([abs(me) for me in (u_ex - S.levels[0].u[-1]) * self.coeff.prefactor])
    else:
        S.levels[0].status.error_extrapolation_estimate = None
