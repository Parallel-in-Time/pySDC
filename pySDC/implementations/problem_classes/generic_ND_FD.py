#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:39:30 2023

@author: telu
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve, cg

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


class GenericNDimFinDiff(ptype):
    r"""
    Base class for finite difference spatial discretisation in :math:`N` dimensions

    .. math::
        \frac{d u}{dt} = A u,

    where :math:`A \in \mathbb{R}^{nN \times nN} is a matrix arising from finite difference discretisation of spatial
    derivatives with :math:`n` degrees of freedom per dimension and :math:`N` dimensions. This generic class follows the MOL
    (method-of-lines) approach and can be used to discretize partial differential equations such as the advection
    equation and the heat equation.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution for the ND problem. For :math:`N = 2`, set ``nvars=(16, 16)``.
    coeff : float, optional
        Factor for finite difference matrix :math:`A`.
    derivative : int, optional
        Order of the spatial derivative.
    freq : tuple of int, optional
        Spatial frequency, can be a tuple.
    stencil_type : str, optional
        Stencil type for finite differences.
    order : int, optional
        Order of accuracy of the finite difference discretization.
    lintol : float, optional
        Tolerance for spatial solver.
    liniter : int, optional
        Maximum number of iterations for linear solver.
    solver_type : str, optional
        Type of solver. Can be ``'direct'``, ``'GMRES'`` or ``'CG'``.
    bc : str, optional
        Type of boundary conditions. Default is ``'periodic'``.

    Attributes
    ----------
    A : sparse matrix (CSC)
        FD discretization matrix of the ND operator.
    Id : sparse matrix (CSC)
        Identity matrix of the same dimension as A.
    xvalues : np.1darray
        Values of spatial grid.
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=512,
        coeff=1.0,
        derivative=1,
        freq=2,
        stencil_type='center',
        order=2,
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
        bc='periodic',
    ):
        # make sure parameters have the correct types
        if not type(nvars) in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')
        if not type(freq) in [int, tuple]:
            raise ProblemError('freq should be either tuple or int')

        # transforms nvars into a tuple
        if type(nvars) is int:
            nvars = (nvars,)

        # automatically determine ndim from nvars
        ndim = len(nvars)
        if ndim > 3:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

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
        for nvar in nvars:
            if nvar % 2 != 0 and bc == 'periodic':
                raise ProblemError('the setup requires nvars = 2^p per dimension')
            if (nvar + 1) % 2 != 0 and bc == 'dirichlet-zero':
                raise ProblemError('setup requires nvars = 2^p - 1')
        if ndim > 1 and nvars[1:] != nvars[:-1]:
            raise ProblemError('need a square domain, got %s' % nvars)

        # invoke super init, passing number of dofs
        super().__init__(init=(nvars[0] if ndim == 1 else nvars, None, np.dtype('float64')))

        # compute dx (equal in both dimensions) and get discretization matrix A
        if bc == 'periodic':
            dx = 1.0 / nvars[0]
            xvalues = np.array([i * dx for i in range(nvars[0])])
        elif bc == 'dirichlet-zero':
            dx = 1.0 / (nvars[0] + 1)
            xvalues = np.array([(i + 1) * dx for i in range(nvars[0])])
        else:
            raise ProblemError(f'Boundary conditions {bc} not implemented.')

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=derivative,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars[0],
            dim=ndim,
            bc=bc,
        )
        self.A *= coeff

        self.xvalues = xvalues
        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('nvars', 'stencil_type', 'order', 'bc', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('freq', 'lintol', 'liniter', 'solver_type', localVars=locals())

        if self.solver_type != 'direct':
            self.work_counters[self.solver_type] = WorkCounter()

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return len(self.nvars)

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
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        return f

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
        solver_type, Id, A, nvars, lintol, liniter, sol = (
            self.solver_type,
            self.Id,
            self.A,
            self.nvars,
            self.lintol,
            self.liniter,
            self.u_init,
        )

        if solver_type == 'direct':
            sol[:] = spsolve(Id - factor * A, rhs.flatten()).reshape(nvars)
        elif solver_type == 'GMRES':
            sol[:] = gmres(
                Id - factor * A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=lintol,
                maxiter=liniter,
                atol=0,
                callback=self.work_counters[solver_type],
                callback_type='legacy',
            )[0].reshape(nvars)
        elif solver_type == 'CG':
            sol[:] = cg(
                Id - factor * A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=lintol,
                maxiter=liniter,
                atol=0,
                callback=self.work_counters[solver_type],
            )[0].reshape(nvars)
        else:
            raise ValueError(f'solver type "{solver_type}" not known in generic advection-diffusion implementation!')

        return sol
