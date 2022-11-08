import numpy as np

from pySDC.core.ProblemDAE import ptype_dae
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh



class pendulum_2d(ptype_dae):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(pendulum_2d, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        g = 9.8
        # The last element of u is a Lagrange multiplier. Not sure if this needs to be time dependent, but must model the
        # weight somehow
        du = u[0:4]
        u = u[5:8]
        f = self.dtype_f(self.init)
        f[:] = (du[0]-u[2],
                     du[1]-u[3],
                     du[2]+u[4]*u[0],
                     du[3]+u[4]*u[1]+g,
                     u[0]**2+u[1]**2-1)
        return f


class simple_dae_1(ptype_dae): 

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(simple_dae_1, self).__init__(problem_params, dtype_u, dtype_f)


    def eval_f(self, u, t):
        # Smooth index-2 DAE pg. 267 Ascher and Petzold (also the first example in KDC Minion paper)
        a = 10.0
        du = u[3:6]
        u = u[0:3]
        f = self.dtype_f(self.init)
        f[:] = (-du[0] + (a - 1 / (2 - t)) * u[0] + (2 - t) * a * u[2] + np.exp(t) * (3 - t) / (2 - t),
                     -du[1] + (1 - a) / (t - 2) * u[0] - u[1] + (a - 1) * u[2] + 2 * np.exp(t),
                     (t + 2) * u[0] + (t ** 2 - 4) * u[1] - (t ** 2 + t - 2) * np.exp(t))
        return f


    def u_exact(self, t):

        me = self.dtype_u(self.init)
        me[:] =(np.exp(t), np.exp(t), -np.exp(t) / (2 - t))
        return me