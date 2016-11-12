from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA


from pySDC.core.Problem import ptype

class heat2d_periodic(ptype):

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        if not 'nu' in problem_params:
            problem_params['nu'] = 1
        if not 'freq' in problem_params:
            problem_params['freq'] = 2
        else:
            assert problem_params['freq'] % 2 == 0, "ERROR: need even number of frequencies due to periodic BCs"

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        assert 'nvars' in problem_params, 'ERROR: need number of nvars for the problem class'
        assert len(problem_params['nvars']) == 2, "ERROR, this is a 2d example, got %s" %problem_params['nvars']
        assert problem_params['nvars'][0] == problem_params['nvars'][1], "ERROR: need a square domain, got %s" %problem_params['nvars']
        assert problem_params['nvars'][0] % 2 == 0, 'ERROR: the setup requires nvars = 2^p per dimension'


        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_periodic, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                       params=problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1 / self.params.nvars[0]
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
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = sp.diags(dstencil,doffsets,shape=(N[0],N[0]), format='csc')
        A = sp.kron(A,sp.eye(N[0])) + sp.kron(sp.eye(N[1]),A)
        A *= nu / (dx**2)

        return A

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
        xvalues = np.array([i*self.dx for i in range(self.params.nvars[0])])
        me.values = np.kron(np.sin(np.pi*self.params.freq*xvalues),np.sin(np.pi*self.params.freq*xvalues)) * \
                    np.exp(-t*self.params.nu*(np.pi*self.params.freq)**2)
        me.values = me.values.flatten()
        return me


