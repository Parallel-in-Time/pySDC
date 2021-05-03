import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.implementations.datatype_classes.mesh import mesh, parallel_imex_mesh


class dynamogp_2d_dedalus(ptype):
    """
    """
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=parallel_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        if 'comm' not in problem_params:
            problem_params['comm'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'Rm', 'kx', 'comm', 'initial']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        ybasis = de.Fourier('y', problem_params['nvars'][0], interval=(0, 2 * np.pi), dealias=1)
        zbasis = de.Fourier('z', problem_params['nvars'][1], interval=(0, 2 * np.pi), dealias=1)
        self.domain = de.Domain([ybasis, zbasis], grid_dtype=np.complex128, comm=problem_params['comm'])

        nvars = tuple(self.domain.local_grid_shape()) + (2,)

        # invoke super init, passing number of dofs (and more), dtype_u and dtype_f
        super(dynamogp_2d_dedalus, self).__init__(init=(nvars, self.domain.dist.comm, ybasis.grid_dtype),
                                                  dtype_u=dtype_u, dtype_f=dtype_f,
                                                  params=problem_params)

        self.y = self.domain.grid(0, scales=1)
        self.z = self.domain.grid(1, scales=1)

        self.problem = de.IVP(domain=self.domain, variables=['by', 'bz'])
        self.problem.parameters['Rm'] = self.params.Rm
        self.problem.parameters['kx'] = self.params.kx
        self.problem.add_equation("dt(by) - (1/Rm) * ( dz(dz(by)) + dy(dy(by)) - kx ** 2 * (by) ) = 0")
        self.problem.add_equation("dt(bz) - (1/Rm) * ( dz(dz(bz)) + dy(dy(bz)) - kx ** 2 * (bz) ) = 0")
        self.solver = self.problem.build_solver(de.timesteppers.SBDF1)
        self.by = self.solver.state['by']
        self.bz = self.solver.state['bz']

        self.u = self.domain.new_field()
        self.v = self.domain.new_field()
        self.w = self.domain.new_field()
        self.w_y = self.domain.new_field()
        self.v_z = self.domain.new_field()

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        f = self.dtype_f(self.init)

        A = np.sqrt(3/2)
        C = np.sqrt(3/2)

        self.u['g'] = A * np.sin(self.z + np.sin(t)) + C * np.cos(self.y + np.cos(t))
        self.v['g'] = A * np.cos(self.z + np.sin(t))
        self.w['g'] = C * np.sin(self.y + np.cos(t))
        self.w_y['g'] = self.w.differentiate(y=1)['g']
        self.v_z['g'] = self.v.differentiate(z=1)['g']

        self.by['g'] = u[..., 0]
        self.bz['g'] = u[..., 1]

        by_y = self.by.differentiate(y=1)
        by_z = self.by.differentiate(z=1)
        bz_y = self.bz.differentiate(y=1)
        bz_z = self.bz.differentiate(z=1)

        tmpfy = (-self.u * self.by * 1j * self.params.kx - self.v * by_y - self.w * by_z +
                 self.bz * self.v_z).evaluate()
        f.expl[..., 0] = tmpfy['g']
        tmpfz = (-self.u * self.bz * 1j * self.params.kx - self.v * bz_y - self.w * bz_z +
                 self.by * self.w_y).evaluate()
        f.expl[..., 1] = tmpfz['g']

        self.by['g'] = u[..., 0]
        self.bz['g'] = u[..., 1]

        pseudo_dt = 1E-05
        self.solver.step(pseudo_dt)

        f.impl[..., 0] = 1.0 / pseudo_dt * (self.by['g'] - u[..., 0])
        f.impl[..., 1] = 1.0 / pseudo_dt * (self.bz['g'] - u[..., 1])

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        self.by['g'] = rhs[..., 0]
        self.bz['g'] = rhs[..., 1]

        self.solver.step(factor)

        me = self.dtype_u(self.init)
        me[..., 0] = self.by['g']
        me[..., 1] = self.bz['g']

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        zvar_loc = self.z.shape[1]
        yvar_loc = self.y.shape[0]

        np.random.seed(0)

        me = self.dtype_u(self.init)

        if self.params.initial == 'random':

            me[..., 0] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))
            me[..., 1] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))

        elif self.params.initial == 'low-res':

            tmp0 = self.domain.new_field()
            tmp1 = self.domain.new_field()
            tmp0['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))
            tmp1['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))

            tmp0.set_scales(4.0 / self.params.nvars[0])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpx = tmp0['g']
            tmp1.set_scales(4.0 / self.params.nvars[1])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpy = tmp1['g']

            tmp0.set_scales(1)
            tmp1.set_scales(1)

            me[..., 0] = tmp0['g']
            me[..., 1] = tmp1['g']

        return me
