from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from playgrounds.deprecated.advection_1d_implicit.getFDMatrix import getFDMatrix
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes import mesh


class advection(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'nvars' in cparams
        assert 'c' in cparams
        assert 'order' in cparams

        assert cparams['nvars']%2 == 0
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advection,self).__init__(self.nvars,dtype_u,dtype_f)

        # compute dx and get discretization matrix A
        self.mesh = np.linspace(0, 1, num=self.nvars, endpoint=False)
        self.dx   = self.mesh[1] - self.mesh[0]
        self.A    = -self.c*getFDMatrix(self.nvars, self.order, self.dx)
    
    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A,rhs.values)
        return me


    def __eval_fimpl(self,u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl        = mesh(self.nvars)
        fimpl.values = self.A.dot(u.values)
        return fimpl


    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = mesh(self.nvars)
        f.values = self.A.dot(u.values)
        return f


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        me = mesh(self.nvars)
        me.values = np.cos(2.0*np.pi*(self.mesh-self.c*t))
        return me
