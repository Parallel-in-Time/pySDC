import numpy as np

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh


class auzinger(ptype):
    """
    Example implementing the van der pol oscillator
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
        assert 'maxiter' in cparams
        assert 'newton_tol' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)
        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super(auzinger,self).__init__(2, dtype_u, dtype_f)


    def u_exact(self,t):
        """
        Dummy routine for the exact solution, currently only passes the initial values

        Args:
            t: current time
        Returns:
            mesh type containing the initial values
        """

        me = mesh(2)
        me.values[0] = np.cos(t)
        me.values[1] = np.sin(t)
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
        f.values[0] = -x2 + x1*(1 - x1**2 - x2**2)
        f.values[1] = x1 + 3*x2*(1 - x1**2 - x2**2)
        return f


    def solve_system(self,rhs,dt,u0):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs: right-hand side for the nonlinear system
            dt: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver

        Returns:
            solution u
        """

        # create new mesh object from u0 and set initial values for iteration
        u = mesh(u0)
        x1 = u.values[0]
        x2 = u.values[1]

        # start newton iteration
        n = 0
        while n < self.maxiter:

            # form the function g with g(u) = 0
            g = np.array([ x1 - dt*(-x2+x1*(1-x1**2-x2**2)) - rhs.values[0], x2 - dt*(x1+3*x2*(1-x1**2-x2**2)) - rhs.values[1] ])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g,np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([ [1-dt*(1-3*x1**2-x2**2), -dt*(-1-2*x1*x2)], [-dt*(1-6*x1*x2), 1-dt*(3-3*x1**2-9*x2**2)] ])

            idg = np.linalg.inv(dg)

            # newton update: u1 = u0 - g/dg
            u.values -= np.dot(idg,g)

            # set new values and increase iteration count
            x1 = u.values[0]
            x2 = u.values[1]
            n += 1

        return u
