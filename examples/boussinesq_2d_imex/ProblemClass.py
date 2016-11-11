import numpy as np
import scipy.sparse.linalg as LA

from build2DFDMatrix import get2DMesh
from buildBoussinesq2DMatrix import getBoussinesq2DMatrix, getBoussinesq2DUpwindMatrix
from pySDC_implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC_core.Problem import ptype
from unflatten import unflatten


class logging(object):

  def __init__(self):
    self.solver_calls = 0
    self.iterations   = 0
    
  def add(self, iterations):
    self.solver_calls += 1
    self.iterations   += iterations
    
class Callback(object):

    def getresidual(self):
      return self.residual
      
    def getcounter(self):
      return self.counter
      
    def __init__(self):
      self.counter=0
      self.residual=0.0
      
    def __call__(self, residuals):
      self.counter+=1
      self.residual=residuals


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
        assert 'order_upw' in cparams
        assert 'order' in cparams
        assert 'gmres_maxiter' in cparams
        assert 'gmres_restart' in cparams
        assert 'gmres_tol_limit' in cparams
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)
        
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(boussinesq_2d_imex,self).__init__(self.nvars,dtype_u,dtype_f)
                
        self.N     = [ self.nvars[1],  self.nvars[2] ]
        
        self.bc_hor = [ ['periodic', 'periodic'] , ['periodic', 'periodic'],   ['periodic', 'periodic'] , ['periodic', 'periodic'] ]
        self.bc_ver = [ ['neumann', 'neumann'] ,   ['dirichlet', 'dirichlet'], ['dirichlet', 'dirichlet'], ['neumann', 'neumann'] ]

        self.xx, self.zz, self.h = get2DMesh(self.N, self.x_bounds, self.z_bounds, self.bc_hor[0], self.bc_ver[0])
       
        self.Id, self.M = getBoussinesq2DMatrix(self.N, self.h, self.bc_hor, self.bc_ver, self.c_s, self.Nfreq, self.order)
        self.D_upwind   = getBoussinesq2DUpwindMatrix( self.N, self.h[0], self.u_adv , self.order_upw)
    
        self.logger = logging()
        self.gmres_tol = None
    
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

        b         = rhs.values.flatten()
        cb        = Callback()

        sol, info = LA.gmres( self.Id - factor*self.M, b, x0=u0.values.flatten(), tol=self.gmres_tol, restart=self.gmres_restart, maxiter=self.gmres_maxiter, callback=cb)
        # If this is a dummy call with factor==0.0, do not log because it should not be counted as a solver call
        if factor!=0.0:
          #print "SDC: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( cb.getcounter(), cb.getresidual() )
          self.logger.add(cb.getcounter())
        me        = mesh(self.nvars)
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
        fexpl        = mesh(self.nvars,val=0.0)
        temp         = u.values.flatten()
        temp         = self.D_upwind.dot(temp)
        fexpl.values = unflatten( temp, 4, self.N[0], self.N[1])
              
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

        temp         = u.values.flatten()
        temp         = self.M.dot(temp)
        fimpl        = mesh(self.nvars,val=0.0)
        fimpl.values = unflatten(temp, 4, self.N[0], self.N[1])
        
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

        f      = rhs_imex_mesh(self.nvars)
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
        H      = 10.0
        a      = 5.0
        x_c    = -50.0
        
        me        = mesh(self.nvars)
        me.values[0,:,:] = 0.0*self.xx
        me.values[1,:,:] = 0.0*self.xx
        #me.values[2,:,:] = 0.0*self.xx
        #me.values[3,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.15**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.15**2)
        #me.values[2,:,:] = np.exp(-0.5*(self.xx-0.0)**2.0/0.05**2.0)*np.exp(-0.5*(self.zz-0.5)**2/0.2**2)
        me.values[2,:,:] = dtheta*np.sin( np.pi*self.zz/H )/( 1.0 + np.square(self.xx - x_c)/(a*a))
        me.values[3,:,:] = 0.0*self.xx
        return me
