import numpy as np

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


# Helper functions
def transistor(u_in):
    return 1e-6 * (np.exp(u_in / 0.026) - 1)


class pendulum_2d(ptype_dae):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(pendulum_2d, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        g = 9.8
        # The last element of u is a Lagrange multiplier. Not sure if this needs to be time dependent, but must model the
        # weight somehow
        du = u[5:10]
        u = u[0:5]
        f = self.dtype_f(self.init)
        f[:] = (du[0]-u[2],
                     du[1]-u[3],
                     du[2]+u[4]*u[0],
                     du[3]+u[4]*u[1]+g,
                     u[0]**2+u[1]**2-1)
        return f

    # dummy exact solution to provide initial conditions for the solver
    def u_exact(self, t): 
        me = self.dtype_u(self.init)
        me[:] = (-1, 0, 0, 0, 0)
        return me


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


# Two transistor amplifier from page 108 Hairer, Lubich and Roche
class two_transistor_amplifier(ptype_dae): 
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(two_transistor_amplifier, self).__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        du = u[8:16]
        u = u[0:8]
        u_b = 6.
        u_e = 0.1 * np.sin(200 * np.pi * t)
        alpha = 0.99
        r_0 = 1000.
        r_k = 9000.
        c_1, c_2, c_3, c_4, c_5 = 1e-6, 2e-6, 3e-6, 4e-6, 5e-6
        f = self.dtype_f(self.init)
        f[:] =((u_e - u[0]) / r_0 - c_1 * (du[0] - du[1]),
                        (u_b - u[1]) / r_k - u[1] / r_k + c_1 * (du[0] - du[1]) + (alpha - 1) * transistor(u[1] - u[2]),
                        transistor(u[1] - u[2]) - u[2] / r_k - c_2 * du[2],
                        (u_b - u[3]) / r_k - c_3 * (du[3] - du[4]) - alpha * transistor(u[1] - u[2]),
                        (u_b - u[4]) / r_k - u[4] / r_k + c_3 * (du[3] - du[4]) + (alpha - 1) * transistor(u[4] - u[5]),
                        transistor(u[4] - u[5]) - u[5] / r_k - c_4 * du[5],
                        (u_b - u[6]) / r_k - c_5 * (du[6] - du[7]) - alpha * transistor(u[4] - u[5]),
                        -u[7] / r_k + c_5 * (du[6] - du[7]))
        return f
        
    def u_exact(self, t):

        me = self.dtype_u(self.init)
        me[:] = (0, 3, 3, 6, 3, 3, 6, 0)
        return me


class problematic_f(ptype_dae):
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super().__init__(problem_params, dtype_u, dtype_f)

    def eval_f(self, u, t):
        eta = 1
        du = u[2:4]
        u = u[0:2]
        f = self.dtype_f(self.init)
        f[:] = (u[0] + eta*t*u[1]-np.sin(t),
                     du[0] + eta*t*du[1] + (1+eta)*u[1]-np.cos(t))
        return f
        
    def u_exact(self, t):

        me = self.dtype_u(self.init)
        me[:] = (0, 0)
        return me