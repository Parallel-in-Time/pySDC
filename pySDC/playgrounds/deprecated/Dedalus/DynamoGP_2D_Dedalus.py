import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class dynamogp_2d_dedalus(ptype):
    """
    """
    def __init__(self, problem_params, dtype_u=dedalus_field, dtype_f=rhs_imex_dedalus_field):
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
        domain = de.Domain([ybasis, zbasis], grid_dtype=np.complex128, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(dynamogp_2d_dedalus, self).__init__(init=(domain, 2), dtype_u=dtype_u, dtype_f=dtype_f,
                                                  params=problem_params)

        self.y = self.init[0].grid(0, scales=1)
        self.z = self.init[0].grid(1, scales=1)

        self.rhs = self.dtype_u(self.init, val=0.0)
        self.problem = de.IVP(domain=self.init[0], variables=['bz', 'by'])
        self.problem.parameters['Rm'] = self.params.Rm
        self.problem.parameters['kx'] = self.params.kx
        self.problem.add_equation("dt(bz) - (1/Rm) * ( dz(dz(bz)) + dy(dy(bz)) - kx ** 2 * (bz) ) = 0")
        self.problem.add_equation("dt(by) - (1/Rm) * ( dz(dz(by)) + dy(dy(by)) - kx ** 2 * (by) ) = 0")
        self.solver = self.problem.build_solver(de.timesteppers.SBDF1)
        self.bz = self.solver.state['bz']
        self.by = self.solver.state['by']

        self.u = domain.new_field()
        self.v = domain.new_field()
        self.w = domain.new_field()
        self.v_z = domain.new_field()
        self.w_y = domain.new_field()

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
        self.v_z['g'] = self.v.differentiate(z=1)['g']
        self.w_y['g'] = self.w.differentiate(y=1)['g']

        by_y = u.values[0].differentiate(y=1)
        by_z = u.values[0].differentiate(z=1)
        bz_y = u.values[1].differentiate(y=1)
        bz_z = u.values[1].differentiate(z=1)

        f.expl.values[0] = (-self.u * u.values[0] * 1j * self.params.kx - self.v * by_y - self.w * by_z +
                            u.values[1] * self.v_z).evaluate()
        f.expl.values[1] = (-self.u * u.values[1] * 1j * self.params.kx - self.v * bz_y - self.w * bz_z +
                            u.values[0] * self.w_y).evaluate()

        f.impl.values[0] = (1.0 / self.params.Rm * (u.values[0].differentiate(z=2) + u.values[0].differentiate(y=2) -
                                                    self.params.kx ** 2 * u.values[0])).evaluate()
        f.impl.values[1] = (1.0 / self.params.Rm * (u.values[1].differentiate(z=2) + u.values[1].differentiate(y=2) -
                                                    self.params.kx ** 2 * u.values[1])).evaluate()

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

        self.bz['g'] = rhs.values[1]['g']
        self.bz['c'] = rhs.values[1]['c']
        self.by['g'] = rhs.values[0]['g']
        self.by['c'] = rhs.values[0]['c']

        self.solver.step(factor)

        me = self.dtype_u(self.init)
        me.values[1]['g'] = self.bz['g']
        me.values[1]['c'] = self.bz['c']
        me.values[0]['g'] = self.by['g']
        me.values[0]['c'] = self.by['c']

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        yvar_loc = self.y.shape[0]
        zvar_loc = self.z.shape[1]

        np.random.seed(0)

        me = self.dtype_u(self.init)

        if self.params.initial == 'random':

            me.values[0]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))
            me.values[1]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))

        elif self.params.initial == 'low-res':

            me.values[0]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))
            me.values[1]['g'] = np.random.uniform(low=-1e-5, high=1e-5, size=(yvar_loc, zvar_loc))

            me.values[0].set_scales(4.0 / self.params.nvars[0])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpx = me.values[0]['g']
            me.values[1].set_scales(4.0 / self.params.nvars[1])
            # Need to do that because otherwise Dedalus tries to be clever..
            tmpy = me.values[1]['g']

            me.values[0].set_scales(1)
            me.values[1].set_scales(1)

        return me
