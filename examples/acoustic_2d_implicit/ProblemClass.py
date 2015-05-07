r"""
  Two-dimensional wave or acoustic equation
  =========================
  
  Integrate the linear 2D acoustic problem:
  
  .. math::
  u_t + c p_x & = 0 \\
  w_t + c p_z &=  0 \\
  p_t + c u_x + c w_z & = 0.
  
  These three equations are the first order system corresponding to the 2D wave equation
  
  ..math::
    p_{tt} + c^2 p_{xx} + c^2 p_{zz} = 0.
    
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

import buildFDMatrix as bfd
from unflatten import unflatten

def u_initial(x):
  return np.exp(-0.5*(x-0.0)**2/0.1**2)


class acoustic_2d_implicit(ptype):
    """
    ....

    Attributes:
      domainx: First output of np.meshgrid
      domainz: Second output of np.meshgrid
      dx: 
      dz:
      Id:
      M:
      Nx:
      Nz:
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
        assert 'cs' in cparams
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(acoustic_2d_implicit,self).__init__(self.nvars,dtype_u,dtype_f)
        
        self.Nx = self.nvars[1]
        self.Nz = self.nvars[2]
        
        x = np.linspace(-3, 3, self.Nx)
        z = np.linspace( 0, 1, self.Nz)
        self.domainx, self.domainz = np.meshgrid(x, z, indexing="ij")
        self.dx     = x[1] - x[0]
        self.dz     = z[1] - z[0]
        
        # Use periodic BC in x direction for u, w, p
        # Use Neumann BC in z direction for u and p
        # Use Dirichlet BC in z direction for w
        
        Ax = bfd.getFDMatrix(self.Nx, self.dx, True)
        Dx = sp.kron( Ax, sp.eye(self.Nz), format="csr")
        
        Az = bfd.getFDMatrix(self.Nz, self.dz, False)
        Az = bfd.modify_delete(Az, 'both')
        Dz = sp.kron( sp.eye(self.Nx), Az, format="csr" )
        
        # Modify the identy matrix to include Neumann BC
        Id_z = sp.eye(self.Nz)
        Id_z = bfd.modify_neumann(Id_z, self.dz, 'both')
        Id   = sp.kron( sp.eye(self.Nx), Id_z,  format="csr")
      
        Zero = sp.csr_matrix(((self.Nx*self.Nz, self.Nx*self.Nz)))
        Id1 = sp.hstack((Id,                      Zero, Zero), format="csr")
        Id2 = sp.hstack((Zero, sp.eye(self.Nx*self.Nz), Zero), format="csr")
        Id3 = sp.hstack((Zero,                    Zero,   Id), format="csr")
        self.Id = sp.vstack((Id1, Id2, Id3), format="csr")
            
        M1  = sp.hstack((Zero, Zero, -Dx), format="csr")
        M2  = sp.hstack((Zero, Zero, -Dz), format="csr")
        M3  = sp.hstack((-Dx,   -Dz, Zero), format="csr")
        self.M = sp.vstack((M1, M2, M3), format="csr")
          
    def solve_system(self,rhs,factor,u0):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """
        
        # Dirichlet BC in z for w and Neumann BC in z for u, p
        for i in range(0,3):
          rhs.values[i,:,0]         = 0.0
          rhs.values[i,:,self.Nz-1] = 0.0
        
        b = rhs.values.flatten()
        
        sol = LA.spsolve( self.Id-factor*self.M, b)
        
        me = mesh(self.nvars)
        me.values = unflatten(sol, 3, self.Nx, self.Nz)
        
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


        fexpl        = mesh(self.nvars)

        # DEBUGGING
        fexpl.values[0,:] = 0.0*self.domainx
        fexpl.values[1,:] = 0.0*self.domainx
        fexpl.values[2,:] = 0.0*self.domainx
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


        b = u.values.flatten()
        sol = self.M.dot(b)
                    
        fimpl = mesh(self.nvars,val=0)
        fimpl.values = unflatten(sol, 3, self.Nx, self.Nz)
        
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
        
        me               = mesh(self.nvars)
        me.values[0,:,:] = 0.0*self.domainx
        me.values[1,:,:] = 0.0*self.domainx
        me.values[2,:,:] = 0.5*u_initial(self.domainx - t) + 0.5*u_initial(self.domainx + t)
        return me


