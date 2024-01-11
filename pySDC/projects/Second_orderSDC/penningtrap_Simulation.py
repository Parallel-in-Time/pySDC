import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN, Velocity_Verlet
from pySDC.projects.Second_orderSDC.plot_helper import PlotManager


class ComputeError(PlotManager):
    """
    This class generates data for plots and computations for Second-order SDC
    """

    def __init__(self, controller_params, description, time_iter=3, K_iter=(1, 2, 3), Tend=2, axes=(1,), cwd=''):
        super().__init__(
            controller_params, description, time_iter=time_iter, K_iter=K_iter, Tend=Tend, axes=axes, cwd=''
        )

    def run_local_error(self):
        """
        Controls everything to generate local convergence rate
        """
        self.compute_local_error_data()
        self.plot_convergence()

    def run_global_error(self):
        """
        Computes global convergence order and finds approximate order
        """
        self.error_type = 'global'
        self.compute_global_error_data()
        self.find_approximate_order(filename='data/dt_vs_global_errorSDC.csv')
        self.plot_convergence()

    def run_work_precision(self, RK=True, VV=False, dt_cont=1):
        """
        Implements work-precision of Second-order SDC
        Args:
            RK: True or False to include RKN method
            VV: True or False to include Velocity-Verlet Scheme
            dt_cont: moves RK and VV left to right
        """
        self.RK = RK
        self.VV = VV
        self.compute_global_error_data(work_counter=True)
        self.compute_global_error_data(Picard=True, work_counter=True)
        if self.RK:
            self.compute_global_error_data(RK=RK, work_counter=True, dt_cont=dt_cont)
        if self.VV:
            self.compute_global_error_data(VV=VV, work_counter=True, dt_cont=dt_cont)
        self.plot_work_precision()

    def compute_local_error_data(self):
        """
        Computes local convergence rate and saves the data
        """
        step_params = dict()
        dt_val = self.description['level_params']['dt']

        for order in self.K_iter:
            step_params['maxiter'] = order
            self.description['step_params'] = step_params

            file_path = self.cwd + 'data/dt_vs_local_errorSDC.csv'
            mode = 'w' if order == self.K_iter[0] else 'a'

            with open(file_path, mode) as file:
                if order == self.K_iter[0]:
                    file.write("Time_steps | Order_pos | Abs_error_position | Order_vel | Abs_error_velocity\n")

                for ii in range(0, self.time_iter):
                    dt = dt_val / 2**ii

                    self.description['level_params']['dt'] = dt
                    self.description['level_params'] = self.description['level_params']

                    controller = controller_nonMPI(
                        num_procs=1, controller_params=self.controller_params, description=self.description
                    )

                    t0, Tend = 0.0, dt
                    P = controller.MS[0].levels[0].prob
                    uinit = P.u_init()

                    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                    uex = P.u_exact(Tend)
                    coll_order = controller.MS[0].levels[0].sweep.coll.order
                    order_pos = list(self.local_order_pos(order, coll_order))
                    order_vel = list(self.local_order_vel(order, coll_order))
                    error_pos = list(np.abs((uex - uend).pos).T[0])
                    error_vel = list(np.abs((uex - uend).vel).T[0])

                    dt_omega = dt * self.description['problem_params']['omega_B']
                    file.write(
                        f"{dt_omega}, {', '.join(map(str, order_pos))}, {', '.join(map(str, error_pos))},"
                        f" {', '.join(map(str, order_vel))}, {', '.join(map(str, error_vel))}\n"
                    )

    def compute_global_error_data(self, Picard=False, RK=False, VV=False, work_counter=False, dt_cont=1):
        """
        Computes global convergence data and saves it into the data folder
        Args:
            Picard: bool, Picard iteration computation
            RK: bool, RKN method
            VV: bool, Velocity-Verlet scheme
            work_counter: bool, compute rhs for work precision
            dt_cont: moves the data left to right for RK and VV method
        """
        K_iter = self.K_iter
        name, description = '', self.description

        if Picard:
            name = 'Picard'
            description['sweeper_params']['QI'] = 'PIC'
            description['sweeper_params']['QE'] = 'PIC'
        elif RK:
            K_iter, name, description['sweeper_class'] = (1,), 'RKN', RKN
        elif VV:
            K_iter, name, description['sweeper_class'] = (1,), 'VV', Velocity_Verlet
        else:
            name = 'SDC'

        self.controller_params['hook_class'] = particles_output
        step_params, dt_val = dict(), self.description['level_params']['dt']
        values, error = ['position', 'velocity'], dict()

        filename = f"data/{'rhs_eval_vs_global_error' if work_counter else 'dt_vs_global_error'}{name}.csv"

        for order in K_iter:
            u_val, uex_val = dict(), dict()
            step_params['maxiter'], description['step_params'] = order, step_params

            file_path = self.cwd + filename
            mode = 'w' if order == K_iter[0] else 'a'

            with open(file_path, mode) as file:
                if order == K_iter[0]:
                    file.write(
                        "Time_steps/Work_counter | Order_pos | Abs_error_position | Order_vel | Abs_error_velocity\n"
                    )

                cont = 2 if self.time_iter == 3 else 2 ** abs(3 - self.time_iter)
                cont = cont if not Picard else dt_cont

                for ii in range(0, self.time_iter):
                    dt = (dt_val * cont) / 2**ii

                    description['level_params']['dt'] = dt
                    description['level_params'] = self.description['level_params']

                    controller = controller_nonMPI(
                        num_procs=1, controller_params=self.controller_params, description=description
                    )

                    t0, Tend = 0.0, self.Tend
                    P = controller.MS[0].levels[0].prob
                    uinit = P.u_init()
                    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                    func_eval = P.work_counters['Boris_solver'].niter + P.work_counters['rhs'].niter

                    for nn in values:
                        u_val[nn] = get_sorted(stats, type=nn, sortby='time')
                        uex_val[nn] = get_sorted(stats, type=nn + '_exact', sortby='time')
                        error[nn] = self.relative_error(uex_val[nn], u_val[nn])
                        error[nn] = list(error[nn].T[0])

                    if RK or VV:
                        global_order = np.array([4, 4])
                    else:
                        coll_order = controller.MS[0].levels[0].sweep.coll.order
                        global_order = list(self.global_order(order, coll_order))
                    global_order = np.array([4, 4]) if RK or VV else list(self.global_order(order, coll_order))

                    dt_omega = dt * self.description['problem_params']['omega_B']
                    save = func_eval if work_counter else dt_omega

                    file.write(
                        f"{save}, {', '.join(map(str, global_order))}, {', '.join(map(str, error['position']))},"
                        f" {', '.join(map(str, global_order))}, {', '.join(map(str, error['velocity']))}\n"
                    )

    def local_order_pos(self, order_K, order_quad):
        if self.description['sweeper_params']['initial_guess'] == 'spread':
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 2 + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 2 + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            else:
                raise NotImplementedError('Order of convergence explicitly not implemented')
        else:
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            else:
                raise NotImplementedError('Order of convergence explicitly not implemented')

    def local_order_vel(self, order_K, order_quad):
        if self.description['sweeper_params']['initial_guess'] == 'spread':
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 1 + 2, order_quad]), np.min([2 * order_K + 2, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 1 + 2, order_quad]), np.min([2 * order_K + 2, order_quad])])
            else:
                raise NotImplementedError('Order of convergence explicitly not implemented')
        else:
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 1, order_quad]), np.min([2 * order_K + 2, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 1, order_quad]), np.min([2 * order_K + 2, order_quad])])
            else:
                raise NotImplementedError('Order of convergence explicitly not implemented')

    def global_order(self, order_K, order_quad):
        if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
            return np.array([np.min([order_K, order_quad]), np.min([2 * order_K, order_quad])])
        elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
            return np.array([np.min([order_K, order_quad]), np.min([2 * order_K, order_quad])]) + 2
        else:
            raise NotImplementedError('Order of convergence explicitly not implemented')

    def relative_error(self, uex_data, u_data):
        u_ex = np.array([entry[1] for entry in uex_data])
        u = np.array([entry[1] for entry in u_data])
        return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)
