from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sLA
import scipy.linalg as LA

from pySDC.Problem import ptype


class advection1d(ptype):
    """
    Example implementing the unforced 1D advection equation with periodic BC in [0,1], discretized using upwinding finite differences

    Attributes:
        A: FD discretization of the gradient operator using upwinding
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
        assert 'c' in problem_params, 'ERROR: need advection parameter for the problem class'
        assert 'freq' in problem_params, 'ERROR: need frequency parameter for the problem class'

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        assert (problem_params['nvars'])%2 == 0, 'ERROR: the setup requires nvars = 2^p'

        if not 'order' in problem_params:
            problem_params['order'] = 1

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advection1d,self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1.0/self.params.nvars
        self.A = self.__get_A(self.params.nvars, self.params.c, self.dx, self.params.order)

    @staticmethod
    def __get_A(N, c, dx, order):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N: number of dofs
            c: diffusion coefficient
            dx: distance between two spatial nodes
            order: specifies order of discretization

        Returns:
         matrix A in CSC format
        """

        coeff = None
        stencil = None
        zero_pos = None

        if order == 1:
            stencil = [-1.0, 1.0]
            coeff = 1.0
            zero_pos = 2

        elif order == 2:
            stencil = [1.0, -4.0, 3.0]
            coeff = 1.0 / 2.0
            zero_pos = 3

        elif order == 3:
            stencil = [1.0, -6.0, 3.0, 2.0]
            coeff = 1.0 / 6.0
            zero_pos = 3

        elif order == 4:
            stencil = [-5.0, 30.0, -90.0, 50.0, 15.0]
            coeff = 1.0 / 60.0
            zero_pos = 4

        elif order == 5:
            stencil = [3.0, -20.0, 60.0, -120.0, 65.0, 12.0]
            coeff = 1.0 / 60.0
            zero_pos = 5
        else:
            print("Order " + order + " not implemented.")
            exit()

        first_col = np.zeros(N)

        # Because we need to specific first column (not row) in circulant, flip stencil array
        first_col[0:np.size(stencil)] = np.flipud(stencil)

        # Circulant shift of coefficient column so that entry number zero_pos becomes first entry
        first_col = np.roll(first_col, -np.size(stencil) + zero_pos, axis=0)

        return sp.csc_matrix(c * coeff * (1.0 / dx) * LA.circulant(first_col))


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
        f.values = -self.A.dot(u.values)
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
        me.values = sLA.spsolve(sp.eye(self.params.nvars)+factor*self.A,rhs.values)
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
        xvalues = np.array([i*self.dx for i in range(self.params.nvars)])
        me.values = np.sin(np.pi*self.params.freq*(xvalues - self.params.c*t))
        return me


