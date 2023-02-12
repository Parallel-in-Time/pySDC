#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:39:30 2023

@author: telu
"""
import numpy as np
import scipy.sparse as sp

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


class GenericNDimFinDiff(ptype):
    
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
        direct_solver=True,
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
        super().__init__(init=(nvars, None, np.dtype('float64')))

        # compute dx (equal in both dimensions) and get discretization matrix A
        if bc == 'periodic':
            xvalues = np.linspace(0, 1, num=nvars[0], endpoint=False)
        elif bc == 'dirichlet-zero':
            xvalues = np.linspace(0, 1, num=nvars[0] + 2)[1:-1]
        else:
            raise ProblemError(f'Boundary conditions {self.params.bc} not implemented.')
        dx = xvalues[1] - xvalues[0]

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
        self._makeAttributeAndRegister('freq', 'lintol', 'liniter', 'direct_solver', localVars=locals())

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
        Routine to evaluate the RHS

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        return f
