from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from scipy.fftpack import fft, ifft
from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

class heat1d(ptype):
    """
    Example implementing the forced 1D heat equation with Periodic boundaries and using fft for
    for the solver in space. This will only work in the interval [0,1]

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
        assert 'nu' in cparams

        assert (cparams['nvars'])%2 == 0

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d,self).__init__(self.nvars, dtype_u, dtype_f)
        N = self.nvars
        self.k = 1j*np.concatenate([np.arange(N/2+1), -1*np.arange(N/2-1)[::-1]-1])
        self.lap = self.k*self.k
        # compute dx and get discretization matrix A
        self.dx = 1/(self.nvars)
        # self.A = self.__get_A(self.nvars,self.nu,self.dx)


    def __get_A(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A in sparse format
        Not needed here but left because of nostalgic reasons
        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil,[-1,0,1],shape=(N,N))
        A = sp.lil_matrix(A)
        A[0,-1] = stencil[0]
        A[-1,0] = stencil[-1]
        A *= nu / (dx**2)
        return A.tocsc()


    def solve_system(self,rhs,factor,u0,t):
        """
        Simple fft solver for (I-dtA)u = rhs
        by computing rhs_fft / ( I - factor * nu * k**2)
        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        # me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A,rhs.values)
        rhs_fft = fft(rhs.values)
        fft_faktor_inv = np.ones(self.nvars, dtype=np.complex128)-factor*self.nu * self.lap
        me.values = ifft(rhs_fft / fft_faktor_inv)
        return me


    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        # xvalues = np.array([(i+1)*self.dx for i in range(self.nvars)])
        fexpl = mesh(self.nvars)
        fexpl.values = np.zeros(self.nvars)#-np.sin(np.pi*xvalues)*(np.sin(t)-self.nu*np.pi**2*np.cos(t))
        return fexpl

    def __eval_fimpl(self,u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl = mesh(self.nvars)
        u_fft = fft(u.values)
        fimpl.values = ifft(self.nu*self.lap*u_fft)
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

        f = rhs_imex_mesh(self.nvars)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
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
        xvalues = np.array([(i)*self.dx for i in range(self.nvars)])
        me.values = np.sin(2*np.pi*xvalues)*np.exp(-t*(2*np.pi)**2*self.nu)
        return me
