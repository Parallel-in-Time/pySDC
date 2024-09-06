from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.spectral_helper import SpectralHelper
import numpy as np
from pySDC.core.errors import ParameterError


class GenericSpectralLinear(Problem):
    """
    Generic class to solve problems of the form M u_t + L u = y, with mass matrix M, linear operator L and some right hand side y using spectral methods.
    L may contain algebraic conditions, as long as (M + dt L) is invertible.

    """

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh
        from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

        cls.xp = cp

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
        solver_type='direct',
        solver_args=None,
        useGPU=False,
        *args,
        **kwargs,
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
        """
        self.spectral = SpectralHelper(comm=comm, useGPU=useGPU)

        if useGPU:
            self.setup_GPU()

        for base in bases:
            self.spectral.add_axis(**base)
        self.spectral.add_component(components)

        self.spectral.setup_fft()

        super().__init__(init=self.spectral.init)

        self.solver_type = solver_type
        self.solver_args = {} if solver_args is None else solver_args

        self.work_counters[solver_type] = WorkCounter()

        self.setup_preconditioner(Dirichlet_recombination, left_preconditioner)

    def __getattr__(self, name):
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
        Dx = self.get_self.get_differentiation_matrix(axes=(0,))
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
        self.diff_index = list(LHS.keys())
        self.diff_mask = [me in self.diff_index for me in self.components]
        self.M = self._setup_operator(LHS)

    def setup_preconditioner(self, Dirichlet_recombination=True, left_preconditioner=True):
        """
        Get left and right precondioners. A right preconditioner of D2T will result in Dirichlet recombination. 10/10 would recommend!

        Args:
            right_preconditioning (str): Basis conversion for right precondioner
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
        Solve (M + dt*L)u=rhs. This requires that you setup the operators before using the functions ``GenericSpectralLinear.setup_L`` and ``GenericSpectralLinear.setup_M``. Note that the mass matrix need not be invertible, as long as (M + dt*L) is. This allows to solve some differential algebraic equations.

        Note that in implicit Euler, the right hand side will be composed of the initial conditions. We don't want that in lines that don't depend on time. Therefore, we multiply the right hand side by the mass matrix. This means you can only do algebraic conditions that add up to zero. But you can easily overload this function with something more generic if needed.

        We use a tau method to enforce boundary conditions in Chebychov methods. This means we replace a line in the system matrix by the polynomials evaluated at a boundary and put the value we want there in the rhs at the respective position. Since we have to do that in spectral space along only the axis we want to apply the boundary condition to, we transform back to real space after applying the mass matrix, and then transform only along one axis, apply the boundary conditions and transform back. Then we transform along all dimensions again. If you desire speed, you may wish to overload this function with something less generic that avoids a few transformations.
        """
        if dt == 0:
            return rhs

        sp = self.spectral.sparse_lib

        sol = self.u_init

        rhs_hat = self.spectral.transform(rhs)
        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(rhs_hat.shape)
        rhs = self.spectral.itransform(rhs_hat).real

        rhs = self.spectral.put_BCs_in_rhs(rhs)
        rhs_hat = self.Pl @ self.spectral.transform(rhs).flatten()

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

        if self.solver_type.lower() == 'direct':
            sol_hat = sp.linalg.spsolve(A, rhs_hat)
        elif self.solver_type.lower() == 'gmres':
            sol_hat, _ = sp.linalg.gmres(
                A,
                rhs_hat,
                x0=u0.flatten(),
                **self.solver_args,
                callback=self.work_counters[self.solver_type],
                callback_type='legacy',
            )
        elif self.solver_type.lower() == 'cg':
            sol_hat, _ = sp.linalg.cg(
                A, rhs_hat, x0=u0.flatten(), **self.solver_args, callback=self.work_counters[self.solver_type]
            )
        else:
            raise NotImplementedError(f'Solver {self.solver_type:!} not implemented in {type(self).__name__}!')

        sol_hat = (self.Pr @ sol_hat).reshape(sol.shape)
        if skip_itransform:
            return sol_hat

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
