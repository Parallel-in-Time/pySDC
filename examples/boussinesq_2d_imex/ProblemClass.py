import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from build2DFDMatrix import get2DMesh
from buildBoussinesq2DMatrix import getBoussinesq2DMatrix, getBoussinesqBCHorizontal, getBoussinesqBCVertical, getBoussinesq2DUpwindMatrix
from unflatten import unflatten


class boussinesq_2d_imex(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
      solver: Sharpclaw solver
      state:  Sharclaw state
      domain: Sharpclaw domain
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
        assert 'c_s' in cparams
        assert 'u_adv' in cparams
        assert 'Nfreq' in cparams
        assert 'x_bounds' in cparams
        assert 'z_bounds' in cparams
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(boussinesq_2d_imex,self).__init__(self.nvars,dtype_u,dtype_f)
                
        self.N     = [ self.nvars[1],  self.nvars[2] ]
        
        self.bc_hor = [ ['periodic', 'periodic'] , ['periodic', 'periodic'],   ['periodic', 'periodic'] , ['periodic', 'periodic'] ]
        self.bc_ver = [ ['neumann', 'neumann'] ,   ['dirichlet', 'dirichlet'], ['dirichlet', 'dirichlet'], ['neumann', 'neumann'] ]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.x_bounds, self.z_bounds, self.bc_hor[0], self.bc_ver[0])
       
        self.Id, self.M = getBoussinesq2DMatrix(self.N, self.h, self.bc_hor, self.bc_ver, self.c_s, self.Nfreq)
        self.D_upwind   = getBoussinesq2DUpwindMatrix( self.N, self.h[0], self.u_adv )
    
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

        b = rhs.values.flatten()
        # NOTE: A = -M, therefore solve Id + factor*M here
        sol, info =  LA.gmres( self.Id + factor*self.M, b, x0=u0.values.flatten(), tol=1e-13, restart=10, maxiter=20)
        me = mesh(self.nvars)
        me.values = unflatten(sol, 4, self.N[0], self.N[1])

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
        
        # Evaluate right hand side
        fexpl = mesh(self.nvars)
        temp  = u.values.flatten()
        temp  = self.D_upwind.dot(temp)
        # NOTE: M_adv = -D_upwind, therefore add a minus here
        fexpl.values = unflatten(-temp, 4, self.N[0], self.N[1])
              
        #fexpl.values = np.zeros((3, self.N[0], self.N[1]))
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

        temp = u.values.flatten()
        temp = self.M.dot(temp)
        fimpl = mesh(self.nvars,val=0)
        # NOTE: M = -A, therefore add a minus here
        fimpl.values = unflatten(-temp, 4, self.N[0], self.N[1])
        
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
        
        dtheta = 0.01
        H      = 10
        a      = 5
        x_c    = 150
        
        me        = mesh(self.nvars)
        me.values[0,:,:] = 0.0*self.xx
        me.values[1,:,:] = 0.0*self.xx
        #me.values[2,:,:] = 0.0*self.xx
        #me.values[3,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.15**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.15**2)
        #me.values[2,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.05**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.2**2)
        me.values[2,:,:] = dtheta*np.sin( np.pi*self.zz/H )/( 1.0 + np.square(self.xx - x_c)/(a*a))
        me.values[3,:,:] = 0.0*self.xx
        return me
