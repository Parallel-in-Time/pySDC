"""
PyTorch helpers for step 7 part D: a tensor datatype, a heat equation problem using it and a (too) simple network.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers import problem_helper

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Tensor(torch.Tensor):
    """
    Wrapper for PyTorch tensor.
    Be aware that this is totally WIP! Should be fine to count iterations, but desperately needs cleaning up if this
    project goes much further!

    TODO: Have to update `torch/multiprocessing/reductions.py` in order to share this datatype across processes.

    Attributes:
        comm: MPI communicator or None
    """

    comm = None

    @staticmethod
    def __new__(cls, init, val=0.0, *args, **kwargs):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a tensor.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        # TODO: The cloning of tensors going in is likely slow

        if isinstance(init, torch.Tensor):
            obj = super().__new__(cls, init.clone())
            obj[:] = init[:]
        elif (
            isinstance(init, tuple)
            and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            # and isinstance(init[2], np.dtype)
        ):
            if isinstance(init[0][0], torch.Tensor):
                obj = super().__new__(cls, init[0].clone())
            else:
                obj = super().__new__(cls, *init[0])
            obj.fill_(val)
            cls.comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(torch.amax(torch.abs(self)))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = 0.0
                global_absval = max(self.comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return float(global_absval)

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.Issend(self[:], dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        return comm.Irecv(self[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self[:], root=root)
        return self


class HeatEquationModel(nn.Module):
    """
    Very simple model to learn the heat equation. Beware! It's too simple.
    Some machine learning expert please fix this!
    """

    def __init__(self, problem, hidden_size=64):
        self.input_size = problem.nvars * 3
        self.output_size = problem.nvars
        self.problem = problem

        super().__init__()

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)

        # Initialize weights (example)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, t, dt):
        # prepare individual tensors
        x = x.float()
        _t = torch.ones(x.shape) * dt
        _dt = torch.ones(x.shape) * dt

        # Concatenate t and dt with the input x
        _x = torch.cat((x, _t, _dt), dim=0)

        _x = self.fc1(_x)
        _x = self.relu(_x)
        _x = self.fc2(_x)
        return _x

    def __call__(self, *args, **kwargs):
        me = self.problem.u_init
        me[:] = super().__call__(*args, **kwargs)
        return me


class Heat1DFDTensor(Problem):
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
        if type(nvars) not in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')
        if type(freq) not in [int, tuple]:
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
