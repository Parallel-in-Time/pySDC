# import matplotlib

# matplotlib.use('Agg')
# import os


import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import convergence_data
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN, Velocity_Verlet
from pySDC.core.Errors import ProblemError
from pySDC.core.Step import step


class Convergence(object):
    """
    Implementation of convergence plot for the Second order SDC
    """

    def __init__(self, controller_params, description, time_iter=3, K_iter=(1, 2, 3), Tend=2, axes=(1,), cwd=''):
        self.controller_params = controller_params
        self.description = description
        self.time_iter = time_iter
        self.K_iter = K_iter
        self.Tend = Tend
        self.axes = axes
        self.cwd = cwd
        self.quad_type = self.description['sweeper_params']['quad_type']
        self.num_nodes = self.description['sweeper_params']['num_nodes']

    # run local convergence rate and plot the graph
    @property
    def run_local_error(self):
        self.error_type = 'Local'
        self.compute_local_error_data()
        self.find_approximate_order()
        self.plot_convergence()

    # run global convergence rate and plot the graph
    @property
    def run_global_error(self):
        self.error_type = 'Global'
        self.compute_global_error_data()
        self.find_approximate_order(filename='data/Global-conv-data.txt')
        self.plot_convergence()

    """
    Plot convergence order plots for the position and velocity
    """

    def plot_convergence(self):
        fs = 16
        [N, time_data, error_data, order_data, convline] = self.organize_data(
            filename='data/{}-conv-data.txt'.format(self.error_type)
        )

        color = ['r', 'brown', 'g', 'blue']
        shape = ['o', 'd', 's', 'x']
        rcParams['figure.figsize'] = 7.44, 6.74
        rcParams['pgf.rcfonts'] = False
        rcParams['xtick.labelsize'] = fs
        rcParams['ytick.labelsize'] = fs
        rcParams['mathtext.fontset'] = 'cm'
        plt.rc('font', size=16)
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        value = self.axes[0]
        for ii in range(0, N):
            ax1.loglog(time_data[ii, :], convline['pos'][value, ii, :], color='black', markersize=fs - 2, linewidth=3)
            ax1.loglog(
                time_data[ii, :],
                error_data['pos'][value, ii, :],
                ' ',
                color=color[ii],
                marker=shape[ii],
                markersize=fs - 2,
                label='k={}'.format(int(self.K_iter[ii])),
            )
            if value == 2:
                ax1.text(
                    time_data[ii, 1],
                    0.3 * convline['pos'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii, 0, 1]),
                    size=fs + 2,
                )
            else:
                ax1.text(
                    time_data[ii, 1],
                    0.3 * convline['pos'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii, 0, 0]),
                    size=fs + 2,
                )

            if self.error_type == 'Local':
                ax1.set_ylabel('$\Delta x^{\mathrm{(abs)}}_{%d}$' % (value + 1), fontsize=fs + 5)
            else:
                ax1.set_ylabel('$\Delta x^{\mathrm{(rel)}}_{%d}$' % (value + 1), fontsize=fs + 5)
        ax1.set_title('{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes), fontsize=fs + 5)
        ax1.set_xlabel('$\omega_{B} \cdot \Delta t$', fontsize=fs + 5)

        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(self.cwd + 'data/{}_conv_plot_pos{}.pdf'.format(self.error_type, value + 1))

        for ii in range(0, N):
            ax2.loglog(time_data[ii, :], convline['vel'][value, ii, :], color='black', markersize=fs - 2, linewidth=3)
            ax2.loglog(
                time_data[ii, :],
                error_data['vel'][value, ii, :],
                ' ',
                color=color[ii],
                marker=shape[ii],
                markersize=fs - 2,
                label='k={}'.format(int(self.K_iter[ii])),
            )

            if value == 2:
                ax2.text(
                    time_data[ii, 1],
                    0.3 * convline['vel'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0, 1]),
                    size=fs + 2,
                )
            else:
                ax2.text(
                    time_data[ii, 1],
                    0.3 * convline['vel'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0, 0]),
                    size=fs + 2,
                )

            if self.error_type == 'Local':
                ax2.set_ylabel('$\Delta v^{\mathrm{(abs)}}_{%d}$' % (value + 1), fontsize=fs + 5)
            else:
                ax2.set_ylabel('$\Delta v^{\mathrm{(rel)}}_{%d}$' % (value + 1), fontsize=fs + 5)
        ax2.set_title('{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes), fontsize=fs + 5)
        ax2.set_xlabel('$\omega_{B} \cdot \Delta t$', fontsize=fs + 5)
        if self.error_type == 'Global':
            ax2.set_ylim(1e-14, 1e1)
            ax1.set_ylim(1e-14, 1e1)
        else:
            ax2.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
            ax1.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        ax2.legend(loc='best')
        fig2.tight_layout()
        fig2.savefig(self.cwd + 'data/{}_conv_plot_vel{}.pdf'.format(self.error_type, value + 1))
        plt.show()

    """
    Compute local convergece data and save it
    """

    def compute_local_error_data(self):
        step_params = dict()
        dt_val = self.description['level_params']['dt']

        # Error storage. It is only for the test
        error_test = dict()
        error_test['pos'] = dict()
        error_test['vel'] = dict()
        error_test['time'] = dict()
        error = dict()
        for order in self.K_iter:
            # define storage for the local error
            # error={'pos': np.zeros([1,3]), 'vel': np.zeros([1,3])}
            error['pos'] = np.zeros([3, self.time_iter])
            error['vel'] = np.zeros([3, self.time_iter])

            step_params['maxiter'] = order
            self.description['step_params'] = step_params

            if order == self.K_iter[0]:
                file = open(self.cwd + 'data/Local-conv-data.txt', 'w')
            else:
                file = open(self.cwd + 'data/Local-conv-data.txt', 'a')

            for ii in range(0, self.time_iter):
                dt = dt_val / 2**ii

                self.description['level_params']['dt'] = dt
                self.description['level_params'] = self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(
                    num_procs=1, controller_params=self.controller_params, description=self.description
                )

                # set time parameters
                t0 = 0.0
                Tend = dt

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                # compute exact solution and compare
                uex = P.u_exact(Tend)

                # find order of quadrature rule
                coll_order = controller.MS[0].levels[0].sweep.coll.order

                # find order of convergence for the postion and velocity
                order_pos = self.local_order_pos(order, coll_order)
                order_vel = self.local_order_vel(order, coll_order)
                # evaluate error
                error['pos'][:, ii] = np.abs((uex - uend).pos).T
                error['vel'][:, ii] = np.abs((uex - uend).vel).T
                if order == self.K_iter[0]:
                    error_test['pos'][dt] = error['pos']
                    error_test['vel'][dt] = error['vel']
                dt_omega = dt * self.description['problem_params']['omega_B']
                file.write(
                    str(dt_omega)
                    + " * "
                    + str(order_pos)
                    + " * "
                    + str(error['pos'][:, ii])
                    + " * "
                    + str(order_vel)
                    + " * "
                    + str(error['vel'][:, ii])
                    + '\n'
                )

            file.close()

    """

    Compute global convergence data and save it data folder

    """

    def compute_global_error_data(self):
        convergence_data.Tend = self.Tend
        self.controller_params['hook_class'] = convergence_data
        step_params = dict()
        dt_val = self.description['level_params']['dt']

        values = ['position', 'velocity']

        error = dict()

        for order in self.K_iter:
            # define storage for the global error
            error['position'] = np.zeros([3, self.time_iter])
            error['velocity'] = np.zeros([3, self.time_iter])
            u_val = dict()
            uex_val = dict()

            step_params['maxiter'] = order
            self.description['step_params'] = step_params

            if order == self.K_iter[0]:
                file = open(self.cwd + 'data/Global-conv-data.txt', 'w')
            else:
                file = open(self.cwd + 'data/Global-conv-data.txt', 'a')

            for ii in range(0, self.time_iter):
                dt = dt_val / 2**ii

                self.description['level_params']['dt'] = dt
                self.description['level_params'] = self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(
                    num_procs=1, controller_params=self.controller_params, description=self.description
                )

                # set time parameters
                t0 = 0.0
                # Tend = dt
                Tend = self.Tend

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                # extract values from stats
                extract_stats = filter_stats(stats, type="error")
                sortedlist_stats = sort_stats(extract_stats, sortby="time")

                sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
                sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel
                # sort values and compute error
                for _, nn in enumerate(values):
                    data = sortedlist_stats[0][1][nn].values()
                    u_val[nn] = np.array(list(data))
                    u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                    data = sortedlist_stats[0][1][nn + "_exact"].values()
                    uex_val[nn] = np.array(list(data))
                    uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                    error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])
                coll_order = controller.MS[0].levels[0].sweep.coll.order
                global_order = self.global_order(order, coll_order)
                dt_omega = dt * self.description['problem_params']['omega_B']

                file.write(
                    str(dt_omega)
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['position'][:, ii])
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['velocity'][:, ii])
                    + '\n'
                )
            file.close()

    # find expected local convergence order for position
    def local_order_pos(self, order_K, order_quad):
        if self.description['sweeper_params']['initial_guess'] == 'spread':
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 2 + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 2 + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            else:
                raise NotImplementedError('order of convergence explicitly not implemented ')
        else:
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 2, order_quad]), np.min([2 * order_K + 3, order_quad])])
            else:
                raise NotImplementedError('order of convergence explicitly not implemented ')

    # find expected local convergence order for velocity
    def local_order_vel(self, order_K, order_quad):
        if self.description['sweeper_params']['initial_guess'] == 'spread':
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 1 + 2, order_quad]), np.min([2 * order_K + 2, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 1 + 2, order_quad]), np.min([2 * order_K + 2, order_quad])])
            else:
                raise NotImplementedError('order of convergence explicitly not implemented ')
        else:
            if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
                return np.array([np.min([order_K + 1, order_quad]), np.min([2 * order_K + 2, order_quad])])
            elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
                return np.array([np.min([order_K + 1, order_quad]), np.min([2 * order_K + 2, order_quad])])
            else:
                raise NotImplementedError('order of convergence explicitly not implemented ')

    # find expected global convergence order
    def global_order(self, order_K, order_quad):
        if self.quad_type == 'GAUSS' or self.quad_type == 'RADAU-RIGHT':
            return np.array([np.min([order_K, order_quad]), np.min([2 * order_K, order_quad])])
        elif self.quad_type == 'LOBATTO' or self.quad_type == 'RADAU-LEFT':
            return np.array([np.min([order_K, order_quad]), np.min([2 * order_K, order_quad])]) + 2
        else:
            raise NotImplementedError('order of convergence explicitly not implemented ')

    # compute relative error
    def relative_error(self, u_ex, u):
        return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)

    # convert string saved data into numpy array
    def string_to_array(self, string):
        numbers = string.strip('[]').split()
        array = [float(num) for num in numbers]
        return np.array(array)

    """

    Seperate data to plot for the graph

    """

    def organize_data(self, filename='data/Local-conv-data.txt', time_iter=None):
        """
        Organize data according to plot
        Args:
            filename (string): data to find approximate order
        """
        if time_iter == None:
            time_iter = self.time_iter

        time = np.array([])
        order = {'pos': np.array([]).reshape([0, 2]), 'vel': np.array([]).reshape([0, 2])}
        error = {'pos': np.array([]).reshape([0, 3]), 'vel': np.array([]).reshape([0, 3])}

        file = open(self.cwd + filename, 'r')

        while True:
            line = file.readline()
            if not line:
                break

            items = str.split(line, " * ", 5)
            time = np.append(time, float(items[0]))

            order['pos'] = np.vstack((order['pos'], self.string_to_array(items[1])))
            order['vel'] = np.vstack((order['vel'], self.string_to_array(items[3])))
            error['pos'] = np.vstack((error['pos'], self.string_to_array(items[2])))
            error['vel'] = np.vstack((error['vel'], self.string_to_array(items[4][:-1])))

        N = int(np.size(time) / time_iter)

        error_data = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}
        order_data = {'pos': np.zeros([N, time_iter, 2]), 'vel': np.zeros([N, time_iter, 2])}
        time_data = np.zeros([N, time_iter])
        convline = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}

        time_data = time.reshape([N, time_iter])

        order_data['pos'][:, :, 0] = order['pos'][:, 0].reshape([N, time_iter])
        order_data['pos'][:, :, 1] = order['pos'][:, 1].reshape([N, time_iter])
        order_data['vel'][:, :, 0] = order['vel'][:, 0].reshape([N, time_iter])
        order_data['vel'][:, :, 1] = order['vel'][:, 1].reshape([N, time_iter])

        for ii in range(0, 3):
            error_data['pos'][ii, :, :] = error['pos'][:, ii].reshape([N, time_iter])
            error_data['vel'][ii, :, :] = error['vel'][:, ii].reshape([N, time_iter])

        for jj in range(0, 3):
            if jj == 2:
                convline['pos'][jj, :, :] = (
                    (time_data / time_data[0, 0]).T ** order_data['pos'][:, jj, 1]
                ).T * error_data['pos'][jj, :, 0][:, None]
                convline['vel'][jj, :, :] = (
                    (time_data / time_data[0, 0]).T ** order_data['vel'][:, jj, 1]
                ).T * error_data['vel'][jj, :, 0][:, None]
            else:
                convline['pos'][jj, :, :] = (
                    (time_data / time_data[0, 0]).T ** order_data['pos'][:, jj, 0]
                ).T * error_data['pos'][jj, :, 0][:, None]
                convline['vel'][jj, :, :] = (
                    (time_data / time_data[0, 0]).T ** order_data['vel'][:, jj, 0]
                ).T * error_data['vel'][jj, :, 0][:, None]

        return [N, time_data, error_data, order_data, convline]

    # find approximate order
    def find_approximate_order(self, filename='data/Local-conv-data.txt'):
        [N, time_data, error_data, order_data, convline] = self.organize_data(self.cwd + filename)
        approx_order = {'pos': np.zeros([1, N]), 'vel': np.zeros([1, N])}

        for jj in range(0, 3):
            if jj == 0:
                file = open(self.cwd + 'data/{}_order_vs_approxorder.txt'.format('Local'), 'w')
            else:
                file = open(self.cwd + 'data/{}_order_vs_approxorder.txt'.format('Local'), 'a')

            for ii in range(0, N):
                approx_order['pos'][0, ii] = np.polyfit(
                    np.log(time_data[ii, :]), np.log(error_data['pos'][jj, ii, :]), 1
                )[0].real
                approx_order['vel'][0, ii] = np.polyfit(
                    np.log(time_data[ii, :]), np.log(error_data['vel'][jj, ii, :]), 1
                )[0].real
            if jj == 2:
                file.write(
                    str(order_data['pos'][:, jj, 1])
                    + ' * '
                    + str(approx_order['pos'][0])
                    + ' * '
                    + str(order_data['vel'][:, jj, 1])
                    + ' * '
                    + str(approx_order['vel'][0])
                    + '\n'
                )
            else:
                file.write(
                    str(order_data['pos'][:, jj, 0])
                    + ' * '
                    + str(approx_order['pos'][0])
                    + ' * '
                    + str(order_data['vel'][:, jj, 0])
                    + ' * '
                    + str(approx_order['vel'][0])
                    + '\n'
                )
        file.close()


class Work_precision(Convergence):

    """
    Implementation Work precision.

    """

    def __init__(
        self,
        controller_params,
        description,
        time_iter=3,
        K_iter=(1, 2, 3),
        Tend=2,
        axes=(0,),
        RKN=True,
        VV=True,
        cwd='',
    ):
        self.RKN = RKN
        self.VV = VV

        super().__init__(
            controller_params, description, time_iter=time_iter, K_iter=K_iter, Tend=Tend, axes=axes, cwd=cwd
        )

    """
    All of the implementations can be controlled in here
    """

    @property
    def run_work_precision(self):
        self.func_eval_SDC()
        self.func_eval_Picard()
        if self.RKN:
            self.func_eval_RKN()
        if self.VV:
            self.func_eval_Velocity_Verlet()
        self.plot_work_precision()

    """
    Compute RHS evalutations for the second order SDC method and save into data folder
    """

    def func_eval_SDC(self):
        convergence_data.Tend = self.Tend
        self.controller_params['hook_class'] = convergence_data
        step_params = dict()
        dt_val = self.description['level_params']['dt']

        values = ['position', 'velocity']
        time_iter = self.time_iter
        error = dict()

        for order in self.K_iter:
            # define storage for the global error
            error['position'] = np.zeros([3, time_iter])
            error['velocity'] = np.zeros([3, time_iter])
            u_val = dict()
            uex_val = dict()

            step_params['maxiter'] = order
            self.description['step_params'] = step_params

            if order == self.K_iter[0]:
                file = open(
                    self.cwd + 'data/func_eval_vs_error_SDC{}{}.txt'.format(self.time_iter, self.num_nodes), 'w'
                )
            else:
                file = open(
                    self.cwd + 'data/func_eval_vs_error_SDC{}{}.txt'.format(self.time_iter, self.num_nodes), 'a'
                )

            for ii in range(0, time_iter):
                dt = (dt_val) / 2**ii

                self.description['level_params']['dt'] = dt
                self.description['level_params'] = self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(
                    num_procs=1, controller_params=self.controller_params, description=self.description
                )

                # set time parameters
                t0 = 0.0
                # Tend = dt
                Tend = self.Tend

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                func_eval = P.work_counters['Boris_solver'].niter + P.work_counters['rhs'].niter
                # extract values from stats
                extract_stats = filter_stats(stats, type="error")
                sortedlist_stats = sort_stats(extract_stats, sortby="time")

                sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
                sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel
                # sort values and compute error
                for _, nn in enumerate(values):
                    data = sortedlist_stats[0][1][nn].values()
                    u_val[nn] = np.array(list(data))
                    u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                    data = sortedlist_stats[0][1][nn + "_exact"].values()
                    uex_val[nn] = np.array(list(data))
                    uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                    error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])
                coll_order = controller.MS[0].levels[0].sweep.coll.order
                global_order = self.global_order(order, coll_order)
                # dt_omega=dt*self.description['problem_params']['omega_B']

                file.write(
                    str(func_eval)
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['position'][:, ii])
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['velocity'][:, ii])
                    + '\n'
                )
            file.close()

    """
    Compute RHS evalutations for the Picard iteration and save into data folder
    """

    def func_eval_Picard(self):
        convergence_data.Tend = self.Tend
        description = self.description
        description['sweeper_params']['QI'] = 'PIC'
        description['sweeper_params']['QE'] = 'PIC'
        self.controller_params['hook_class'] = convergence_data
        step_params = dict()
        dt_val = description['level_params']['dt']

        values = ['position', 'velocity']

        error = dict()

        for order in self.K_iter:
            # define storage for the global error
            error['position'] = np.zeros([3, self.time_iter])
            error['velocity'] = np.zeros([3, self.time_iter])
            u_val = dict()
            uex_val = dict()

            step_params['maxiter'] = order
            description['step_params'] = step_params

            if order == self.K_iter[0]:
                file = open(
                    self.cwd + 'data/func_eval_vs_error_picard{}{}.txt'.format(self.time_iter, self.num_nodes), 'w'
                )
            else:
                file = open(
                    self.cwd + 'data/func_eval_vs_error_picard{}{}.txt'.format(self.time_iter, self.num_nodes), 'a'
                )

            # Controller for plot
            if self.time_iter == 3:
                cont = 2
            else:
                tt = np.abs(3 - self.time_iter)
                cont = 2**tt + 2

            for ii in range(0, self.time_iter):
                dt = (dt_val * cont) / 2**ii

                description['level_params']['dt'] = dt
                description['level_params'] = self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(
                    num_procs=1, controller_params=self.controller_params, description=description
                )

                # set time parameters
                t0 = 0.0
                # Tend = dt
                Tend = self.Tend

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
                func_eval = P.work_counters['rhs'].niter
                # extract values from stats
                extract_stats = filter_stats(stats, type="error")
                sortedlist_stats = sort_stats(extract_stats, sortby="time")

                sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
                sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel
                # sort values and compute error
                for _, nn in enumerate(values):
                    data = sortedlist_stats[0][1][nn].values()
                    u_val[nn] = np.array(list(data))
                    u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                    data = sortedlist_stats[0][1][nn + "_exact"].values()
                    uex_val[nn] = np.array(list(data))
                    uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                    error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])
                coll_order = controller.MS[0].levels[0].sweep.coll.order
                global_order = self.global_order(order, coll_order)
                # dt_omega=dt*self.description['problem_params']['omega_B']

                file.write(
                    str(func_eval)
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['position'][:, ii])
                    + " * "
                    + str(global_order)
                    + " * "
                    + str(error['velocity'][:, ii])
                    + '\n'
                )
            file.close()

    """
    Compute RHS evalutations for the Runge-Kutta-Nystrom method and save into data folder
    """

    def func_eval_RKN(self):
        convergence_data.Tend = self.Tend
        description = self.description
        self.controller_params['hook_class'] = convergence_data
        step_params = dict()
        dt_val = description['level_params']['dt']
        # description['level_params'].pop('restol')

        values = ['position', 'velocity']

        error = dict()

        # define storage for the global error
        error['position'] = np.zeros([3, self.time_iter])
        error['velocity'] = np.zeros([3, self.time_iter])
        u_val = dict()
        uex_val = dict()
        order = 1
        step_params['maxiter'] = order
        description['step_params'] = step_params
        description['sweeper_class'] = RKN

        file = open(self.cwd + 'data/func_eval_vs_error_RKN{}{}.txt'.format(self.time_iter, self.num_nodes), 'w')

        # Controller for plot
        if self.time_iter == 3:
            cont = 1
        else:
            tt = np.abs(3 - self.time_iter)
            cont = (1 / 4) ** tt

        for ii in range(0, self.time_iter):
            dt = (dt_val * cont) / 2**ii

            description['level_params']['dt'] = dt
            description['level_params'] = self.description['level_params']

            # instantiate the controller (no controller parameters used here)
            controller = controller_nonMPI(
                num_procs=1, controller_params=self.controller_params, description=description
            )

            # set time parameters
            t0 = 0.0
            # Tend = dt
            Tend = self.Tend

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_init()

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            func_eval = P.work_counters['rhs'].niter

            # extract values from stats
            extract_stats = filter_stats(stats, type="error")
            sortedlist_stats = sort_stats(extract_stats, sortby="time")

            sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
            sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel

            # sort values and compute error
            for _, nn in enumerate(values):
                data = sortedlist_stats[0][1][nn].values()
                u_val[nn] = np.array(list(data))
                u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                data = sortedlist_stats[0][1][nn + "_exact"].values()
                uex_val[nn] = np.array(list(data))
                uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])

            global_order = np.array([4, 4])
            # dt_omega=dt*self.description['problem_params']['omega_B']

            file.write(
                str(func_eval)
                + " * "
                + str(global_order)
                + " * "
                + str(error['position'][:, ii])
                + " * "
                + str(global_order)
                + " * "
                + str(error['velocity'][:, ii])
                + '\n'
            )
        file.close()

    """
    Compute RHS evalutations for the Velocity-Verlet scheme method and save into data folder
    """

    def func_eval_Velocity_Verlet(self):
        convergence_data.Tend = self.Tend
        description = self.description
        self.controller_params['hook_class'] = convergence_data
        step_params = dict()
        dt_val = description['level_params']['dt']
        # description['level_params'].pop('restol')

        values = ['position', 'velocity']

        error = dict()

        # define storage for the global error
        error['position'] = np.zeros([3, self.time_iter])
        error['velocity'] = np.zeros([3, self.time_iter])
        u_val = dict()
        uex_val = dict()

        step_params['maxiter'] = 1
        description['step_params'] = step_params
        description['sweeper_class'] = Velocity_Verlet

        # if order ==self.K_iter[0]:
        file = open(self.cwd + 'data/func_eval_vs_error_VV{}{}.txt'.format(self.time_iter, self.num_nodes), 'w')

        # Controller for plot
        if self.time_iter == 3:
            cont = 1
        else:
            tt = np.abs(3 - self.time_iter)
            cont = 8**tt + 8

        for ii in range(0, self.time_iter):
            dt = (dt_val * cont) / 2**ii

            description['level_params']['dt'] = dt
            description['level_params'] = self.description['level_params']

            # instantiate the controller (no controller parameters used here)
            controller = controller_nonMPI(
                num_procs=1, controller_params=self.controller_params, description=description
            )

            # set time parameters
            t0 = 0.0
            # Tend = dt
            Tend = self.Tend

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_init()

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            func_eval = P.work_counters['rhs'].niter
            # extract values from stats
            extract_stats = filter_stats(stats, type="error")
            sortedlist_stats = sort_stats(extract_stats, sortby="time")

            sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
            sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel

            # sort values and compute error
            for _, nn in enumerate(values):
                data = sortedlist_stats[0][1][nn].values()
                u_val[nn] = np.array(list(data))
                u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                data = sortedlist_stats[0][1][nn + "_exact"].values()
                uex_val[nn] = np.array(list(data))
                uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])

            global_order = np.array([2, 2])

            file.write(
                str(func_eval)
                + " * "
                + str(global_order)
                + " * "
                + str(error['position'][:, ii])
                + " * "
                + str(global_order)
                + " * "
                + str(error['velocity'][:, ii])
                + '\n'
            )
        file.close()

    def format_number(self, data_value, indx):
        if data_value >= 1_000_000:
            formatter = "{:1.1f}M".format(data_value * 0.000_001)
        else:
            formatter = "{:1.0f}K".format(data_value * 0.001)
        return formatter

    """
    Plot work precision from the saved datas
    """

    def plot_work_precision(self):
        fs = 16
        [N, func_eval_SDC, error_SDC, *_] = self.organize_data(
            filename=self.cwd + 'data/func_eval_vs_error_SDC{}{}.txt'.format(self.time_iter, self.num_nodes),
            time_iter=self.time_iter,
        )

        [N, func_eval_picard, error_picard, *_] = self.organize_data(
            filename=self.cwd + 'data/func_eval_vs_error_picard{}{}.txt'.format(self.time_iter, self.num_nodes),
            time_iter=self.time_iter,
        )

        color = ['r', 'brown', 'g', 'blue']
        shape = ['o', 'd', 's', 'x']
        rcParams['figure.figsize'] = 7.44, 6.74
        rcParams['pgf.rcfonts'] = False
        rcParams['xtick.labelsize'] = fs
        rcParams['ytick.labelsize'] = fs
        rcParams['mathtext.fontset'] = 'cm'
        plt.rc('font', size=15)
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        value = self.axes[0]

        if self.RKN:
            [N, func_eval_RKN, error_RKN, *_] = self.organize_data(
                filename=self.cwd + 'data/func_eval_vs_error_RKN{}{}.txt'.format(self.time_iter, self.num_nodes),
                time_iter=self.time_iter,
            )

            ax1.loglog(
                func_eval_RKN[0][1:],
                error_RKN['pos'][value,][0][:][1:],
                ls='dashdot',
                color='purple',
                marker='p',
                label='RKN-4',
                markersize=fs - 3,
            )
            ax2.loglog(
                func_eval_RKN[0][1:],
                error_RKN['vel'][value,][0][:][1:],
                ls='dashdot',
                color='purple',
                marker='p',
                label='RKN-4',
                markersize=fs - 3,
            )
        if self.VV:
            [N, func_eval_VV, error_VV, *_] = self.organize_data(
                filename=self.cwd + 'data/func_eval_vs_error_VV{}{}.txt'.format(self.time_iter, self.num_nodes),
                time_iter=self.time_iter,
            )

            ax1.loglog(
                func_eval_VV[0],
                error_VV['pos'][value,][0][:],
                ls='dashdot',
                color='blue',
                marker='H',
                label='Velocity-Verlet',
                markersize=fs - 3,
            )
            ax2.loglog(
                func_eval_VV[0],
                error_VV['vel'][value,][0][:],
                ls='dashdot',
                color='blue',
                marker='H',
                label='Velocity-Verlet',
                markersize=fs - 3,
            )

        for ii, jj in enumerate(self.K_iter):
            """For the third axis plot uncomment this and set time_iter=4"""
            # if ii==0 or ii==1:
            #     ax1.loglog(func_eval_SDC[ii, :][1:], error_SDC['pos'][value, ii, :][1:], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj), markersize=fs-3)
            #     ax1.loglog(func_eval_picard[ii,:][1:], error_picard['pos'][value, ii, :][1:], ls='--', color=color[ii], marker=shape[ii], markersize=fs-3)

            #     ax2.loglog(func_eval_SDC[ii, :][1:], error_SDC['vel'][value, ii, :][1:], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj), markersize=fs-3)
            #     ax2.loglog(func_eval_picard[ii,:][1:], error_picard['vel'][value, ii, :][1:], ls='--', color=color[ii], marker=shape[ii], markersize=fs-3)
            # else:

            #     ax1.loglog(func_eval_SDC[ii, :][:-1], error_SDC['pos'][value, ii, :][:-1], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj), markersize=fs-3)
            #     ax1.loglog(func_eval_picard[ii,:][:-1], error_picard['pos'][value, ii, :][:-1], ls='--', color=color[ii], marker=shape[ii], markersize=fs-3)

            #     ax2.loglog(func_eval_SDC[ii, :][:-1], error_SDC['vel'][value, ii, :][:-1], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj), markersize=fs-3)
            #     ax2.loglog(func_eval_picard[ii,:][:-1], error_picard['vel'][value, ii, :][:-1], ls='--', color=color[ii], marker=shape[ii], markersize=fs-3)

            ax1.loglog(
                func_eval_SDC[ii, :],
                error_SDC['pos'][value, ii, :],
                ls='solid',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(jj),
                markersize=fs - 2,
            )
            ax1.loglog(
                func_eval_picard[ii, :],
                error_picard['pos'][value, ii, :],
                ls='--',
                color=color[ii],
                marker=shape[ii],
                markersize=fs - 2,
            )

            ax2.loglog(
                func_eval_SDC[ii, :],
                error_SDC['vel'][value, ii, :],
                ls='solid',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(jj),
                markersize=fs - 2,
            )
            ax2.loglog(
                func_eval_picard[ii, :],
                error_picard['vel'][value, ii, :],
                ls='--',
                color=color[ii],
                marker=shape[ii],
                markersize=fs - 2,
            )

        xmin = np.min(ax1.get_xlim())
        xmax = np.max(ax2.get_xlim())
        xmin = round(xmin, -3)
        xmax = round(xmax, -3)

        xx = np.linspace(np.log(xmin), np.log(xmax), 5)
        xx = 3**xx
        xx = xx[np.where(xx < xmax)]
        # xx=[2*1e+3,4*1e+3, 8*1e+3]
        ax1.grid(True)

        ax1.set_title("$M={}$".format(self.num_nodes), fontsize=fs - 2)
        ax1.set_xlabel("Number of RHS evaluations", fontsize=fs + 4)
        ax1.set_ylabel('$\Delta x^{\mathrm{(rel)}}_{%d}$' % (value + 1), fontsize=fs + 5)
        ax1.loglog([], [], color="black", ls="--", label="Picard iteration")
        ax1.loglog([], [], color="black", ls="solid", label="Boris-SDC iteration")

        ax1.set_xticks(xx)
        ax1.xaxis.set_major_formatter(self.format_number)
        ax1.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        # ax1.set_ylim(1e-14, 1e+0)

        ax1.legend(loc="best", fontsize=fs - 4)
        fig1.tight_layout()
        fig1.savefig(self.cwd + "data/f_eval_pos_{}_M={}.pdf".format(value, self.num_nodes))

        ax2.grid(True)
        ax2.xaxis.set_major_formatter(self.format_number)
        ax2.set_title("$M={}$".format(self.num_nodes), fontsize=fs - 2)
        ax2.set_xlabel("Number of RHS evaluations", fontsize=fs + 3)
        ax2.set_ylabel('$\Delta v^{\mathrm{(rel)}}_{%d}$' % (value + 1), fontsize=fs + 3)
        ax2.loglog([], [], color="black", ls="--", label="Picard iteration")
        ax2.loglog([], [], color="black", ls="solid", label="Boris-SDC iteration")
        ax2.set_xticks(xx)
        ax2.xaxis.set_major_formatter(self.format_number)
        ax2.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        # ax2.set_ylim(1e-14, 1e+0)
        ax2.legend(loc="best", fontsize=fs - 3)
        fig2.tight_layout()
        fig2.savefig(self.cwd + "data/f_eval_vel_{}_M={}.pdf".format(value, self.num_nodes))

        plt.show()


class Stability_implementation(object):
    """
    Get necessary values for the computation of stability function and store them.
    """

    def __init__(self, description, kappa_max=20, mu_max=20, Num_iter=(400, 400), cwd=''):
        self.description = description
        self.kappa_max = kappa_max
        self.mu_max = mu_max
        self.kappa_iter = Num_iter[0]
        self.mu_iter = Num_iter[1]
        self.lambda_kappa = np.linspace(0.0, self.kappa_max, self.kappa_iter)
        self.lambda_mu = np.linspace(0.0, self.mu_max, self.mu_iter)
        self.K_iter = description['step_params']['maxiter']
        self.num_nodes = description['sweeper_params']['num_nodes']
        self.dt = description['level_params']['dt']
        self.SDC, self.Ksdc, self.picard, self.Kpicard = self.stability_data()
        self.cwd = cwd

    """
    Routine to compute the stability domains of different configurations of fwsw-SDC

    Returns:
        numpy.ndarray: domain_SDC
        numpy.ndarray: domain_Ksdc
        numpy.ndarray: domain_picard
        numpy.ndarray: domain_Kpicard
    """

    def stability_data(self):
        S = step(description=self.description)

        L = S.levels[0]

        Q = L.sweep.coll.Qmat[1:, 1:]
        QQ = np.dot(Q, Q)
        num_nodes = L.sweep.coll.num_nodes
        dt = L.params.dt
        Q_coll = np.block([[QQ, np.zeros([num_nodes, num_nodes])], [np.zeros([num_nodes, num_nodes]), Q]])
        qQ = np.dot(L.sweep.coll.weights, Q)

        ones = np.block([[np.ones(num_nodes), np.zeros(num_nodes)], [np.zeros(num_nodes), np.ones(num_nodes)]])

        q_mat = np.block(
            [
                [dt**2 * qQ, np.zeros(num_nodes)],
                [np.zeros(num_nodes), dt * L.sweep.coll.weights],
            ]
        )

        domain_SDC = np.zeros((self.kappa_iter, self.mu_iter), dtype="complex")
        domain_picard = np.zeros((self.kappa_iter, self.mu_iter))
        domain_Ksdc = np.zeros((self.kappa_iter, self.mu_iter))
        domain_Kpicard = np.zeros((self.kappa_iter, self.mu_iter))
        for i in range(0, self.kappa_iter):
            for j in range(0, self.mu_iter):
                k = self.lambda_kappa[i]
                mu = self.lambda_mu[j]
                F = np.block(
                    [
                        [-k * np.eye(num_nodes), -mu * np.eye(num_nodes)],
                        [-k * np.eye(num_nodes), -mu * np.eye(num_nodes)],
                    ]
                )
                if self.K_iter != 0:
                    lambdas = [k, mu]
                    SDC_mat_sweep, Ksdc_eigval = L.sweep.get_scalar_problems_manysweep_mats(
                        nsweeps=self.K_iter, lambdas=lambdas
                    )
                    if L.sweep.params.picard_mats_sweep:
                        (
                            Picard_mat_sweep,
                            Kpicard_eigval,
                        ) = L.sweep.get_scalar_problems_picardsweep_mats(nsweeps=self.K_iter, lambdas=lambdas)
                    else:
                        ProblemError("Picard interation is False")
                    domain_Ksdc[i, j] = Ksdc_eigval
                    if L.sweep.params.picard_mats_sweep:
                        domain_Kpicard[i, j] = Kpicard_eigval

                else:
                    SDC_mat_sweep = np.linalg.inv(np.eye(2 * num_nodes) - dt * np.dot(Q_coll, F))

                if L.sweep.params.do_coll_update:
                    FSDC = np.dot(F, SDC_mat_sweep)
                    Rsdc_mat = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FSDC) @ ones.T
                    stab_func, v = np.linalg.eig(Rsdc_mat)

                    if L.sweep.params.picard_mats_sweep:
                        FPicard = np.dot(F, Picard_mat_sweep)
                        Rpicard_mat = np.array([[1.0, dt], [0, 1.0]]) + np.dot(q_mat, FPicard) @ ones.T
                        stab_func_picard, v = np.linalg.eig(Rpicard_mat)
                else:
                    pass
                    raise ProblemError("Collocation update step is only works for True")

                domain_SDC[i, j] = np.max(np.abs(stab_func))
                if L.sweep.params.picard_mats_sweep:
                    domain_picard[i, j] = np.max(np.abs(stab_func_picard))

        return (
            dt * domain_SDC.real,
            dt * domain_Ksdc.real,
            dt * domain_picard.real,
            dt * domain_Kpicard.real,
        )

    """
    Stability function of RKN method

    Returns:
        float: maximum absolute values of eigvales
    """

    def stability_function_RKN(self, k, mu, dt):
        A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        B = np.array([[0, 0, 0, 0], [0.125, 0, 0, 0], [0.125, 0, 0, 0], [0, 0, 0.5, 0]])
        c = np.array([0, 0.5, 0.5, 1])
        b = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
        bA = np.array([1 / 6, 1 / 6, 1 / 6, 0])
        L = np.eye(4) + k * (dt**2) * B + mu * dt * A
        R = np.block([[-k * np.ones(4)], [-(k * c + mu * np.ones(4))]])

        K = np.linalg.inv(L) @ R.T
        C = np.block([[dt**2 * bA], [dt * b]])
        Y = np.array([[1, dt], [0, 1]]) + C @ K
        eigval = np.linalg.eigvals(Y)

        return np.max(np.abs(eigval))

    def stability_data_RKN(self):
        """
        Compute and store values into a matrix

        Returns:
            numpy.ndarray: stab_RKN
        """
        stab_RKN = np.zeros([self.kappa_iter, self.mu_iter])
        for ii, kk in enumerate(self.lambda_kappa):
            for jj, mm in enumerate(self.lambda_mu):
                stab_RKN[jj, ii] = self.stability_function_RKN(kk, mm, self.dt)

        return stab_RKN

    def plot_stability(self, region, title=""):
        """
        Plotting runtine for moduli

        Args:
            stabval (numpy.ndarray): moduli
            title: title for the plot
        """
        fs = 20
        rcParams['figure.figsize'] = 7.44, 6.74
        rcParams['pgf.rcfonts'] = False
        rcParams['xtick.labelsize'] = fs
        rcParams['ytick.labelsize'] = fs
        rcParams['mathtext.fontset'] = 'cm'
        plt.rc('font', size=15)
        lam_k_max = np.amax(self.lambda_kappa)
        lam_mu_max = np.amax(self.lambda_mu)

        plt.figure()
        levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

        CS1 = plt.contour(
            self.lambda_kappa, self.lambda_mu, np.absolute(region.T), levels, colors='k', linestyles="dashed"
        )
        # CS2 = plt.contour(self.lambda_k, self.lambda_mu, np.absolute(region.T), [1.0], colors='k')

        plt.clabel(CS1, inline=True, fmt="%3.2f", fontsize=fs - 2)

        # plt.xticks(np.arange(0, lam_k_max+2, 2), np.arange(0,int(lam_k_max+2), 2))
        plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 3, 3))
        plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 3, 3))
        plt.gca().tick_params(axis="both", which="both", labelsize=fs)
        plt.xlim([0.0, lam_k_max])
        plt.ylim([0.0, lam_mu_max])
        plt.xlabel(r"$\Delta t\cdot \kappa }$", fontsize=fs + 4, labelpad=0.0)  # \ \mathrm{(Spring \ pendulum)
        plt.ylabel(r"$\Delta t\cdot \mu }$", fontsize=fs + 4, labelpad=0.0)  # \ \mathrm{(Friction)
        if self.RKN:
            plt.title(f"{title}", fontsize=fs)
        else:
            plt.title("{}  $M={},\  K={}$".format(title, self.num_nodes, self.K_iter), fontsize=fs)
        plt.tight_layout()
        plt.savefig(self.cwd + "data/M={}_K={}_redion_{}.pdf".format(self.num_nodes, self.K_iter, title))

    def plot_spec_radius(self, region, title=""):
        """
        Plotting runtine for moduli

        Args:
            stabval (numpy.ndarray): moduli
            title: title for the plot
        """
        fs = 20
        rcParams['figure.figsize'] = 7.44, 6.74
        rcParams['pgf.rcfonts'] = False
        rcParams['xtick.labelsize'] = fs
        rcParams['ytick.labelsize'] = fs
        rcParams['mathtext.fontset'] = 'cm'
        plt.rc('font', size=15)
        lam_k_max = np.amax(self.lambda_kappa)
        lam_mu_max = np.amax(self.lambda_mu)

        plt.figure()
        levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

        CS1 = plt.contour(
            self.lambda_kappa, self.lambda_mu, np.absolute(region.T), levels, colors='k', linestyles="dashed"
        )
        # CS2 = plt.contour(self.lambda_k, self.lambda_mu, np.absolute(region.T), [1.0])

        plt.clabel(CS1, inline=True, fmt="%3.2f", fontsize=fs - 2)

        plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 3, 3))
        plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 3, 3))
        plt.gca().tick_params(axis="both", which="both", labelsize=fs)
        plt.xlim([0.0, lam_k_max])
        plt.ylim([0.0, lam_mu_max])
        plt.xlabel(r"$\Delta t\cdot \kappa }$", fontsize=fs + 4, labelpad=0.0)  # \ \mathrm{(Spring \ pendulum)
        plt.ylabel(r"$\Delta t\cdot \mu }$", fontsize=fs + 4, labelpad=0.0)  # \ \mathrm{(Friction)
        plt.title("{}  $M={}$".format(title, self.num_nodes), fontsize=fs)
        plt.tight_layout()
        plt.savefig(self.cwd + "data/M={}_redion_{}.pdf".format(self.num_nodes, title))

    @property
    def run_SDC_stability(self):
        self.RKN = False
        self.plot_stability(self.SDC, title="SDC stability region")

    @property
    def run_Picard_stability(self):
        self.RKN = False
        self.plot_stability(self.picard, title="Picard stability region")

    @property
    def run_Ksdc(self):
        self.plot_spec_radius(self.Ksdc, title="$K_{sdc}$ spectral radius")

    @property
    def run_Kpicard(self):
        self.plot_spec_radius(self.Kpicard, title="$K_{picard}$ spectral radius")

    @property
    def run_RKN_stability(self):
        self.RKN = True
        region_RKN = self.stability_data_RKN()
        self.plot_stability(region_RKN.T, title='RKN-4 stability region')
