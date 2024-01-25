import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve, cg
import torch
from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.playgrounds.ML_initial_guess.tensor import Tensor
from pySDC.playgrounds.ML_initial_guess.sweeper import GenericImplicitML_IG
from pySDC.tutorial.step_1.A_spatial_problem_setup import run_accuracy_check
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.ML_initial_guess.ml_heat import HeatEquationModel


class Heat1DFDTensor(ptype):
    """
    Very simple 1-dimensional finite differences implementation of a heat equation using the pySDC-PyTorch interface.
    Still includes some mess.
    """

    dtype_u = Tensor
    dtype_f = Tensor

    def __init__(
        self,
        nvars=256,
        nu=1.0,
        freq=4,
        stencil_type='center',
        order=2,
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
        bc='periodic',
        bcParams=None,
    ):
        # make sure parameters have the correct types
        if not type(nvars) in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')
        if not type(freq) in [int, tuple]:
            raise ProblemError('freq should be either tuple or int')

        ndim = 1

        # eventually extend freq to other dimension
        if type(freq) is int:
            freq = (freq,) * ndim
        if len(freq) != ndim:
            raise ProblemError(f'len(freq)={len(freq)}, different to ndim={ndim}')

        # check values for freq and nvars
        for f in freq:
            if ndim == 1 and f == -1:
                # use Gaussian initial solution in 1D
                bc = 'periodic'
                break
            if f % 2 != 0 and bc == 'periodic':
                raise ProblemError('need even number of frequencies due to periodic BCs')

        # invoke super init, passing number of dofs
        super().__init__(init=(torch.empty(size=(nvars,), dtype=torch.double), None, np.dtype('float64')))

        dx, xvalues = problem_helper.get_1d_grid(size=nvars, bc=bc, left_boundary=0.0, right_boundary=1.0)

        self.A_, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars,
            dim=ndim,
            bc=bc,
        )
        self.A_ *= nu
        self.A = torch.tensor(self.A_.todense())

        self.xvalues = torch.tensor(xvalues, dtype=torch.double)
        self.Id = torch.tensor((sp.eye(nvars, format='csc')).todense())

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('nvars', 'stencil_type', 'order', 'bc', 'nu', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('freq', 'lintol', 'liniter', 'solver_type', localVars=locals())

        if self.solver_type != 'direct':
            self.work_counters[self.solver_type] = WorkCounter()

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return 1

    @property
    def dx(self):
        """Size of the mesh (in all dimensions)"""
        return self.xvalues[1] - self.xvalues[0]

    @property
    def grids(self):
        """ND grids associated to the problem"""
        x = self.xvalues
        if self.ndim == 1:
            return x
        if self.ndim == 2:
            return x[None, :], x[:, None]
        if self.ndim == 3:
            return x[None, :, None], x[:, None, None], x[None, None, :]

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Values of the right-hand side of the problem.
        """
        f = self.f_init
        f[:] = torch.matmul(self.A, u)
        return f

    def ML_predict(self, u0, t0, dt):
        """
        Predict the solution at t0+dt given initial conditions u0
        """
        # read in model
        model = HeatEquationModel(self)
        model.load_state_dict(torch.load('heat_equation_model.pth'))
        model.eval()

        # evaluate model
        predicted_state = model(u0, t0, dt)
        sol = self.u_init
        sol[:] = predicted_state.double()[:]
        return sol

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        sol : dtype_u
            The solution of the linear solver.
        """
        solver_type, Id, A, nvars, sol = (
            self.solver_type,
            self.Id,
            self.A,
            self.nvars,
            self.u_init,
        )

        if solver_type == 'direct':
            sol[:] = torch.linalg.solve(Id - factor * A, rhs.flatten()).reshape(nvars)
        # TODO: implement torch equivalent of cg
        # elif solver_type == 'CG':
        #     sol[:] = cg(
        #         Id - factor * A,
        #         rhs.flatten(),
        #         x0=u0.flatten(),
        #         tol=lintol,
        #         maxiter=liniter,
        #         atol=0,
        #         callback=self.work_counters[solver_type],
        #     )[0].reshape(nvars)
        else:
            raise ValueError(f'solver type "{solver_type}" not known!')

        return sol

    def u_exact(self, t, **kwargs):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        sol : dtype_u
            The exact solution.
        """
        if 'u_init' in kwargs.keys() or 't_init' in kwargs.keys():
            self.logger.warning(
                f'{type(self).__name__} uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
            )

        ndim, freq, nu, dx, sol = self.ndim, self.freq, self.nu, self.dx, self.u_init

        if ndim == 1:
            x = self.grids
            rho = (2.0 - 2.0 * torch.cos(np.pi * freq[0] * dx)) / dx**2
            if freq[0] > 0:
                sol[:] = torch.sin(np.pi * freq[0] * x) * torch.exp(-t * nu * rho)
        else:
            raise NotImplementedError

        return sol


def main():
    """
    A simple test program to setup a full step instance
    """
    dt = 1e-2

    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'NN'

    problem_params = dict()

    step_params = dict()
    step_params['maxiter'] = 20

    description = dict()
    description['problem_class'] = Heat1DFDTensor
    description['problem_params'] = problem_params
    description['sweeper_class'] = GenericImplicitML_IG
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 20}, description=description)

    P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(0)
    uend, _ = controller.run(u0=uinit, t0=0, Tend=dt)
    u_exact = P.u_exact(dt)
    print("error ", torch.abs(u_exact - uend).max())


if __name__ == "__main__":
    main()
