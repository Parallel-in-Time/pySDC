from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh


class vanderpol(ptype):
    """
    Example implementing the van der pol oscillator
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'u0' in cparams
        assert 'mu' in cparams
        assert 'maxiter' in cparams
        assert 'newton_tol' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)
        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super(vanderpol,self).__init__(2, dtype_u, dtype_f)


    def u_exact(self,t):
        """
        Dummy routine for the exact solution, currently only passes the initial values

        Args:
            t: current time
        Returns:
            mesh type containing the initial values
        """

        # thou shall not call this at time > 0
        assert t is 0
        me = mesh(2)
        me.values = self.u0
        return me

    def eval_f(self,u,t):
        """
        Routine to compute the RHS for both components simultaneously

        Args:
            t: current time (not used here)
            u: the current values
        Returns:
            RHS, 2 components
        """

        x1 = u.values[0]
        x2 = u.values[1]
        f = mesh(2)
        f.values[0] = x2
        f.values[1] = self.mu*(1-x1**2)*x2 - x1
        return f


    def solve_system(self,rhs,dt,u0,t):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs: right-hand side for the nonlinear system
            dt: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution u
        """

        mu = self.mu

        # create new mesh object from u0 and set initial values for iteration
        u = mesh(u0)
        x1 = u.values[0]
        x2 = u.values[1]

        # start newton iteration
        n = 0
        while n < self.maxiter:

            # form the function g with g(u) = 0
            g = np.array([ x1 - dt*x2 - rhs.values[0], x2 - dt*(mu*(1-x1**2)*x2-x1) - rhs.values[1] ])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g,np.inf)
            if res < self.newton_tol:
                break

            # prefactor for dg/du
            c = 1.0/(-2*dt**2*mu*x1*x2 - dt**2 - 1 + dt*mu*(1-x1**2))
            # assemble dg/du
            dg = c*np.array([ [dt*mu*(1-x1**2)-1, -dt], [2*dt*mu*x1*x2+dt, -1] ])

            # newton update: u1 = u0 - g/dg
            u.values -= np.dot(dg,g)

            # set new values and increase iteration count
            x1 = u.values[0]
            x2 = u.values[1]
            n += 1

        return u


    def eval_jacobian(self, u):

        x1 = u.values[0]
        x2 = u.values[1]

        dfdu = np.array( [ [0, 1], [-2*self.mu*x1*x2 -1, self.mu*(1-x1**2)] ] )

        return dfdu


    def apply_jacobian(self, dfdu, u):


        dfduxu = mesh(2)
        dfduxu.values = dfdu.dot(u.values)

        return dfduxu

    def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            dfdu: the Jacobian of the RHS of the ODE
            rhs: right-hand side for the linear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = mesh(2)
        me.values = LA.spsolve(sp.eye(2) - factor * dfdu, rhs.values)
        return me