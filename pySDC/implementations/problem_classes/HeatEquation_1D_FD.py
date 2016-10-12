from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA
import logging

from pySDC.Problem import ptype

class heat1d(ptype):
    """
    Example implementing the unforced 1D heat equation with Dirichlet-0 BC in [0,1], discretized using central finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'nvars' in problem_params, 'ERROR: need number of nvars for the problem class'
        assert 'nu' in problem_params, 'ERROR: need diffusion parameter for the problem class'
        assert 'freq' in problem_params, 'ERROR: need frequency parameter for the problem class'

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        assert (problem_params['nvars']+1)%2 == 0, 'ERROR: the setup requires nvars = 2^p-1'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d,self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1/(self.params.nvars + 1)
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx)

    def __get_A(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil,[-1,0,1],shape=(N,N))
        A *= nu / (dx**2)
        return A.tocsc()

    def eval_f(self,u,t):
        """
        Routine to evaluate the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS
        """

        f = self.dtype_f(self.init)
        f.values = self.A.dot(u.values)
        return f

    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs: right-hand side for the linear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = self.dtype_u(self.init)
        me.values = LA.spsolve(sp.eye(self.params.nvars)-factor*self.A,rhs.values)
        return me

    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i+1)*self.dx for i in range(self.params.nvars)])
        me.values = np.sin(np.pi*self.params.freq*xvalues)*np.exp(-t*self.params.nu*(np.pi*self.params.freq)**2)
        return me


