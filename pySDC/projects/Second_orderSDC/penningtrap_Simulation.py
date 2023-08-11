# import matplotlib

# matplotlib.use('Agg')
# import os

import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN, Velocity_Verlet
from pySDC.core.Errors import ProblemError
from pySDC.core.Step import step


def fixed_plot_params():  # pragma: no cover
    """
    Setting fixed parameters for the all of the plots
    """
    fs = 16
    plt.rcParams['figure.figsize'] = 7.44, 6.74
    plt.rcParams['pgf.rcfonts'] = False


    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = fs+5
    plt.rcParams['axes.labelsize'] = fs+5
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['xtick.major.pad'] = 5
    plt.rcParams['ytick.major.pad'] = 5
    plt.rcParams['axes.labelpad'] = 6
    plt.rcParams['lines.markersize'] = fs-2
    plt.rcParams['lines.markeredgewidth'] = 1
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rc('font', size=fs)


class plotmanager(object):  # pragma: no cover
    """
    This class generates all of the plots of the Second-order SDC plots.
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
        self.error_type = 'local'



    def plot_convergence(self):  # pragma: no cover
        """
        Plot convergence order plots for the position and velocity
        If you change parameters of the values you need set y_lim values need to set manually
        """
        fixed_plot_params()
        [N, time_data, error_data, order_data, convline] = self.organize_data(
            filename='data/dt_vs_{}_errorSDC.csv'.format(self.error_type)
        )

        color = ['r', 'brown', 'g', 'blue']
        shape = ['o', 'd', 's', 'x']

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        value = self.axes[0]
        for ii in range(0, N):
            ax1.loglog(time_data[ii, :], convline['pos'][value, ii, :], color='black')
            ax1.loglog(
                time_data[ii, :],
                error_data['pos'][value, ii, :],
                ' ',
                color=color[ii],
                marker=shape[ii],

                label='k={}'.format(int(self.K_iter[ii])),
            )
            if value == 2:
                ax1.text(
                    time_data[ii, 1],
                    0.25 * convline['pos'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii, 0, 1]),
                    size=18
                )
            else:
                ax1.text(
                    time_data[ii, 1],
                    0.25 * convline['pos'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii, 0, 0]),
                    size=18
                )

            if self.error_type == 'Local':
                ax1.set_ylabel(r'$\Delta x^{\mathrm{(abs)}}_{%d}$' % (value + 1))
            else:
                ax1.set_ylabel(r'$\Delta x^{\mathrm{(rel)}}_{%d}$' % (value + 1))
        ax1.set_title('{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes))
        ax1.set_xlabel(r'$\omega_{B} \cdot \Delta t$')

        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(self.cwd + 'data/{}_conv_plot_pos{}.pdf'.format(self.error_type, value + 1))

        for ii in range(0, N):
            ax2.loglog(time_data[ii, :], convline['vel'][value, ii, :], color='black')
            ax2.loglog(
                time_data[ii, :],
                error_data['vel'][value, ii, :],
                ' ',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(int(self.K_iter[ii])),
            )

            if value == 2:
                ax2.text(
                    time_data[ii, 1],
                    0.25 * convline['vel'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0, 1]),
                    size=18
                )
            else:
                ax2.text(
                    time_data[ii, 1],
                    0.25 * convline['vel'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0, 0]),
                    size=18
                )

            if self.error_type == 'Local':
                ax2.set_ylabel(r'$\Delta v^{\mathrm{(abs)}}_{%d}$' % (value + 1))
            else:
                ax2.set_ylabel(r'$\Delta v^{\mathrm{(rel)}}_{%d}$' % (value + 1))
        ax2.set_title(r'{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes))
        ax2.set_xlabel(r'$\omega_{B} \cdot \Delta t$')
# =============================================================================
#       Setting y axis min and max values
# =============================================================================
        if self.error_type == 'Global':
            ax2.set_ylim(1e-14, 1e1)
            ax1.set_ylim(1e-14, 1e1)
        else:
            ax2.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
            ax1.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        ax2.legend(loc='best')
        fig2.tight_layout()
        plt.show()
        fig2.savefig(self.cwd + 'data/{}_conv_plot_vel{}.pdf'.format(self.error_type, value + 1))

    def format_number(self, data_value, indx):  # pragma: no cover
        """
        Change format of the x axis for the work precision plots
        """
        if data_value >= 1_000_000:
            formatter = "{:1.1f}M".format(data_value * 0.000_001)
        else:
            formatter = "{:1.0f}K".format(data_value * 0.001)
        return formatter



    def plot_work_precision(self):  # pragma: no cover
        """
        Generate work precision plots
        """
        fixed_plot_params()
        [N, func_eval_SDC, error_SDC, *_] = self.organize_data(
            filename=self.cwd + 'data/rhs_eval_vs_global_errorSDC.csv',
            time_iter=self.time_iter,
        )

        [N, func_eval_picard, error_picard, *_] = self.organize_data(
            filename=self.cwd + 'data/rhs_eval_vs_global_errorPicard.csv',
            time_iter=self.time_iter,
        )

        color = ['r', 'brown', 'g', 'blue']
        shape = ['o', 'd', 's', 'x']
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        value = self.axes[0]

        if self.RK:
            [N, func_eval_RKN, error_RKN, *_] = self.organize_data(
                filename=self.cwd + 'data/rhs_eval_vs_global_errorRKN.csv',
                time_iter=self.time_iter,
            )

            ax1.loglog(
                func_eval_RKN[0],
                error_RKN['pos'][value,][0][:],
                ls='dashdot',
                color='purple',
                marker='p',
                label='RKN-4',
            )
            ax2.loglog(
                func_eval_RKN[0],
                error_RKN['vel'][value,][0][:],
                ls='dashdot',
                color='purple',
                marker='p',
                label='RKN-4',
            )
        if self.VV:
            [N, func_eval_VV, error_VV, *_] = self.organize_data(
                filename=self.cwd + 'data/rhs_eval_vs_global_errorVV.csv',
                time_iter=self.time_iter,
            )

            ax1.loglog(
                func_eval_VV[0],
                error_VV['pos'][value,][0][:],
                ls='dashdot',
                color='blue',
                marker='H',
                label='Velocity-Verlet'
            )
            ax2.loglog(
                func_eval_VV[0],
                error_VV['vel'][value,][0][:],
                ls='dashdot',
                color='blue',
                marker='H',
                label='Velocity-Verlet'
            )

        for ii, jj in enumerate(self.K_iter):
# =============================================================================
#           # If you want to get exactly the same picture like in paper uncomment this only for vertical axis
            # if ii==0 or ii==1:
            #     ax1.loglog(func_eval_SDC[ii, :][1:], error_SDC['pos'][value, ii, :][1:], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj))
            #     ax1.loglog(func_eval_picard[ii,:][1:], error_picard['pos'][value, ii, :][1:], ls='--', color=color[ii], marker=shape[ii])

            #     ax2.loglog(func_eval_SDC[ii, :][1:], error_SDC['vel'][value, ii, :][1:], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj))
            #     ax2.loglog(func_eval_picard[ii,:][1:], error_picard['vel'][value, ii, :][1:], ls='--', color=color[ii], marker=shape[ii])
            # else:

            #     ax1.loglog(func_eval_SDC[ii, :][:-1], error_SDC['pos'][value, ii, :][:-1], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj))
            #     ax1.loglog(func_eval_picard[ii,:][:-1], error_picard['pos'][value, ii, :][:-1], ls='--', color=color[ii], marker=shape[ii])

            #     ax2.loglog(func_eval_SDC[ii, :][:-1], error_SDC['vel'][value, ii, :][:-1], ls='solid', color=color[ii], marker=shape[ii], label='k={}'.format(jj))
            #     ax2.loglog(func_eval_picard[ii,:][:-1], error_picard['vel'][value, ii, :][:-1], ls='--', color=color[ii], marker=shape[ii])
#
# =============================================================================
            ax1.loglog(
                func_eval_SDC[ii, :],
                error_SDC['pos'][value, ii, :],
                ls='solid',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(jj)
            )
            ax1.loglog(
                func_eval_picard[ii, :],
                error_picard['pos'][value, ii, :],
                ls='--',
                color=color[ii],
                marker=shape[ii]
            )

            ax2.loglog(
                func_eval_SDC[ii, :],
                error_SDC['vel'][value, ii, :],
                ls='solid',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(jj)
            )
            ax2.loglog(
                func_eval_picard[ii, :],
                error_picard['vel'][value, ii, :],
                ls='--',
                color=color[ii],
                marker=shape[ii]
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

        ax1.set_title("$M={}$".format(self.num_nodes))
        ax1.set_xlabel("Number of RHS evaluations")
        ax1.set_ylabel(r'$\Delta x^{\mathrm{(rel)}}_{%d}$' % (value + 1))
        ax1.loglog([], [], color="black", ls="--", label="Picard iteration")
        ax1.loglog([], [], color="black", ls="solid", label="Boris-SDC iteration")

        ax1.set_xticks(xx)
        ax1.xaxis.set_major_formatter(self.format_number)
        ax1.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        # ax1.set_ylim(1e-14, 1e+0)

        ax1.legend(loc="best", fontsize=12)#,
        fig1.tight_layout()
        fig1.savefig(self.cwd + "data/f_eval_pos_{}_M={}.pdf".format(value, self.num_nodes))

        ax2.grid(True)
        ax2.xaxis.set_major_formatter(self.format_number)
        ax2.set_title("$M={}$".format(self.num_nodes))
        ax2.set_xlabel("Number of RHS evaluations")
        ax2.set_ylabel(r'$\Delta v^{\mathrm{(rel)}}_{%d}$' % (value + 1))
        ax2.loglog([], [], color="black", ls="--", label="Picard iteration")
        ax2.loglog([], [], color="black", ls="solid", label="Boris-SDC iteration")
        ax2.set_xticks(xx)
        ax2.xaxis.set_major_formatter(self.format_number)
        ax2.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        # ax2.set_ylim(1e-14, 1e+0)
        ax2.legend(loc="best", fontsize=12)
        fig2.tight_layout()
        fig2.savefig(self.cwd + "data/f_eval_vel_{}_M={}.pdf".format(value, self.num_nodes))
        plt.show()



    def organize_data(self, filename='data/dt_vs_local_errorSDC.csv', time_iter=None):  # pragma: no cover
        """
        Organize data according to plot
        Args:
            filename (string): data to find approximate order
            time_iter : in case it you used different time iterations
        """
        if time_iter == None:
            time_iter = self.time_iter


        items=np.genfromtxt(filename, delimiter=',', skip_header=1)
        time=items[:,0]
        N = int(np.size(time) / time_iter)



        error_data = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}
        order_data = {'pos': np.zeros([N, time_iter, 2]), 'vel': np.zeros([N, time_iter, 2])}
        time_data = np.zeros([N, time_iter])
        convline = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}

        time_data = time.reshape([N, time_iter])

        order_data['pos'][:, :, 0] = items[:, 1].reshape([N, time_iter])
        order_data['pos'][:, :, 1] = items[:, 2].reshape([N, time_iter])
        order_data['vel'][:, :, 0] = items[:, 6].reshape([N, time_iter])
        order_data['vel'][:, :, 1] = items[:, 7].reshape([N, time_iter])

        for ii in range(0, 3):
            error_data['pos'][ii, :, :] = items[:,ii + 3].reshape([N, time_iter])
            error_data['vel'][ii, :, :] = items[:,ii + 8].reshape([N, time_iter])

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
    def find_approximate_order(self, filename='data/dt_vs_local_errorSDC.csv'):
        """
        This function finds approximate convergence rate and saves in the data folder
        Args:
            filename: given data
        return:
            None
        """
        [N, time_data, error_data, order_data, convline] = self.organize_data(self.cwd + filename)
        approx_order = {'pos': np.zeros([1, N]), 'vel': np.zeros([1, N])}

        for jj in range(0, 3):
            if jj == 0:
                file = open(self.cwd + 'data/{}_order_vs_approxorder.csv'.format(self.error_type), 'w')
                file.write(
                    'Expected pos'
                    + ' | '
                    + 'Measured convergence rate for pos'
                    + ' | '
                    'Expected vel'
                    + ' | '
                    + 'Measured convergence rate for vel'
                    + '\n'
                )
            else:
                file = open(self.cwd + 'data/{}_order_vs_approxorder.csv'.format(self.error_type), 'a')

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
                    + ' | '
                    + str(approx_order['pos'][0])
                    + ' | '
                    + str(order_data['vel'][:, jj, 1])
                    + ' | '
                    + str(approx_order['vel'][0])
                    + '\n'
                )
            else:
                file.write(
                    str(order_data['pos'][:, jj, 0])
                    + ' | '
                    + str(approx_order['pos'][0])
                    + ' | '
                    + str(order_data['vel'][:, jj, 0])
                    + ' | '
                    + str(approx_order['vel'][0])
                    + '\n'
                )
        file.close()

    def plot_hamiltonian(self, ham_SDC, ham_RKN):
        tn=np.max(np.shape(ham_RKN))
        time=np.linspace(0, self.Tend, tn)
        import pdb
        pdb.set_trace()
        plt.semilogy(time, ham_RKN[:,1])



class compute_error(plotmanager):
    """
    This class generates the data for the plots and computations for Second-order SDC
    """

    def __init__(self, controller_params, description, time_iter=3, K_iter=(1, 2, 3), Tend=2, axes=(1,), cwd=''):
        super().__init__(controller_params, description, time_iter=time_iter, K_iter=K_iter, Tend=Tend, axes=axes, cwd='')


    def run_local_error(self):  # pragma: no cover
        """
        This function controlls everything to generate local convergence rate
        """
        self.compute_local_error_data()
        # self.find_approximate_order()
        self.plot_convergence()


    def run_global_error(self):  # pragma: no cover
        """
        This function for the global convergence order together it finds approximate order
        """
        self.error_type = 'global'
        self.compute_global_error_data()
        self.find_approximate_order(filename='data/dt_vs_global_errorSDC.csv')
        self.plot_convergence()


    def run_work_precision(self, RK=True, VV=False, dt_cont=3):  # pragma: no cover
        """
        To implement work-precision of Second-order SDC
        Args:
            RK: True or False to include in the picture RKN method
            VV: True or False to include in the picture Velocity-Verlet Scheme
            dt_cont: moves RK and VV left to right (I could't find the best way instead of this)
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

    def run_hamiltonian_error(self):
        Hamiltonian_SDC=self.compute_global_error_data()
        Hamiltonia_RKN=self.compute_error_RKN_VV()
        self.plot_hamiltonian(Hamiltonian_SDC, Hamiltonia_RKN)


    def compute_local_error_data(self):

        """
        Compute local convergece rate and save this data
        """

        step_params = dict()
        dt_val = self.description['level_params']['dt']

        for order in self.K_iter:



            step_params['maxiter'] = order
            self.description['step_params'] = step_params

            if order == self.K_iter[0]:
                file = open(self.cwd + 'data/dt_vs_local_errorSDC.csv', 'w')
                file.write(str('Time_steps')
                + " | "
                + str('Order_pos')
                + " | "
                + str('Abs_error_position')
                + " | "
                + str('Order_vel')
                + " | "
                + str('Abs_error_velocity')
                + '\n'
            )
            else:
                file = open(self.cwd + 'data/dt_vs_local_errorSDC.csv', 'a')

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
                order_pos = list(self.local_order_pos(order, coll_order))
                order_vel = list(self.local_order_vel(order, coll_order))
                # evaluate error
                error_pos=list(np.abs((uex-uend).pos).T[0])
                error_vel=list(np.abs((uex-uend).vel).T[0])


                dt_omega = dt * self.description['problem_params']['omega_B']
                file.write(
                    str(dt_omega)
                    + ", "
                    + str(', '.join(map(str, order_pos)))
                    + ", "
                    + str(', '.join(map(str, error_pos)))
                    + ", "
                    + str(', '.join(map(str, order_vel)))
                    + ", "
                    + str(', '.join(map(str, error_vel)))
                    + '\n'
                )

            file.close()


    def compute_global_error_data(self, Picard=False, RK=False, VV=False, work_counter=False, dt_cont=1):
        """
        Compute global convergence data and save it into the data folder
        Args:
            Picard: bool, Picard iteration computation
            RK: bool, RKN method
            VV: bool, Velocity-Verlet scheme
            work_counter: bool, compute rhs for the work precision
            dt_cont: moves the data left to right for RK and VV method
        """

        K_iter=self.K_iter
        if Picard:
            name = 'Picard'
            description = self.description
            description['sweeper_params']['QI'] = 'PIC'
            description['sweeper_params']['QE'] = 'PIC'

        elif RK:
            K_iter=(1, )
            name='RKN'
            description=self.description
            description['sweeper_class']=RKN
        elif VV:
            K_iter=(1, )
            name='VV'
            description=self.description
            description['sweeper_class']=Velocity_Verlet
        else:
            name = 'SDC'
            description = self.description
        self.controller_params['hook_class'] = particles_output
        step_params = dict()
        dt_val = self.description['level_params']['dt']

        values = ['position', 'velocity']

        error = dict()
        # Hamiltonian=dict()
        if work_counter:
            filename = 'data/rhs_eval_vs_global_error{}.csv'.format(name)
        else:
            filename = 'data/dt_vs_global_error{}.csv'.format(name)

        for order in K_iter:
            u_val = dict()
            uex_val = dict()

            step_params['maxiter'] = order
            description['step_params'] = step_params

            if order == K_iter[0]:
                file = open(self.cwd + filename, 'w')
                file.write(str('Time_steps/Work_counter')
                + " | "
                + str('Order_pos')
                + " | "
                + str('Abs_error_position')
                + " | "
                + str('Order_vel')
                + " | "
                + str('Abs_error_velocity')
                + '\n'
            )
            else:
                file = open(self.cwd + filename, 'a')

            # Controller for plot
            if Picard:
                if self.time_iter == 3:
                    cont = 2
                else:
                    tt = np.abs(3 - self.time_iter)
                    cont = 2**tt + 2
            else:
                cont = dt_cont

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
                # rhs function evaluation
                func_eval = P.work_counters['Boris_solver'].niter + P.work_counters['rhs'].niter
                # extract values from stats
                for _, nn in enumerate(values):
                    u_val[nn]=get_sorted(stats,type=nn, sortby='time')
                    uex_val[nn]=get_sorted(stats,type=nn+'_exact', sortby='time')
                    error[nn]=self.relative_error(uex_val[nn], u_val[nn])
                    error[nn]=list(error[nn].T[0])


                # if ii==0:
                #     Hamiltonian[order]=self.Hamiltonian_error(u_val, uinit)
                if RK or VV:
                    global_order=np.array([4,4])
                else:
                    coll_order = controller.MS[0].levels[0].sweep.coll.order
                    global_order = list(self.global_order(order, coll_order))
                dt_omega = dt * self.description['problem_params']['omega_B']
                if work_counter:
                    save = func_eval
                else:
                    save = dt_omega
                file.write(
                    str(save)
                    + ", "
                    + str(', '.join(map(str, global_order)))
                    + ", "
                    + str(', '.join(map(str, error['position'])))
                    + ", "
                    + str(', '.join(map(str, global_order)))
                    + ", "
                    + str(', '.join(map(str, error['velocity'])))
                    + '\n'
                )
            file.close()

        # return Hamiltonian

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
    def relative_error(self, uex_data, u_data):
        u_ex=np.array([entry[1] for entry in uex_data])
        u=np.array([entry[1] for entry in u_data])
        return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)

    def Hamiltonian_error(self, u, u0):
        shape=np.shape(u['position'])

        Hn=0.5*(u['position']**2+u['velocity']**2)

        u0pos=u0.pos.T*np.ones(shape)
        u0vel=u0.vel.T*np.ones(shape)
        H0=0.5*(u0pos**2+u0vel**2)

        H=np.abs(Hn-H0)/np.abs(H0)

        return H



class Stability_implementation(object):
    """
    Routine to compute the stability domains of different configurations of SDC
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


    def stability_data(self):
        """
        Computes stability domain matrix for the Harmonic oscillator problem
        Returns:
            numpy.ndarray: domain_SDC
            numpy.ndarray: domain_Ksdc
            numpy.ndarray: domain_picard
            numpy.ndarray: domain_Kpicard
        """
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



    def stability_function_RKN(self, k, mu, dt):
        """
        Stability function of RKN method

        Returns:
            float: maximum absolute values of eigvales
        """
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

    def plot_stability(self, region, title=""):  # pragma: no cover
        """
        Plotting runtine for moduli

        Args:
            stabval (numpy.ndarray): moduli
            title: title for the plot
        """
        fixed_plot_params()
        lam_k_max = np.amax(self.lambda_kappa)
        lam_mu_max = np.amax(self.lambda_mu)

        plt.figure()
        levels = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.1])

        CS1 = plt.contour(self.lambda_kappa, self.lambda_mu, region.T, levels, colors='k', linestyles="dashed")
        # CS2 = plt.contour(self.lambda_k, self.lambda_mu, np.absolute(region.T), [1.0], colors='r')

        plt.clabel(CS1, inline=True, fmt="%3.2f")


        plt.gca().set_xticks(np.arange(0, int(lam_k_max) + 3, 3))
        plt.gca().set_yticks(np.arange(0, int(lam_mu_max) + 3, 3))
        plt.gca().tick_params(axis="both", which="both")
        plt.xlim([0.0, lam_k_max])
        plt.ylim([0.0, lam_mu_max])

        plt.xlabel(r"$\Delta t\cdot \kappa }$", labelpad=0.0)
        plt.ylabel(r"$\Delta t\cdot \mu }$", labelpad=0.0)
        if self.RKN:
            plt.title(f"{title}")
        if self.radius:
            plt.title("{}  $M={}$".format(title, self.num_nodes))
        else:
            plt.title(r"{}  $M={},\  K={}$".format(title, self.num_nodes, self.K_iter))
        plt.tight_layout()
        plt.savefig(self.cwd + "data/M={}_K={}_redion_{}.pdf".format(self.num_nodes, self.K_iter, title))


    def run_SDC_stability(self):  # pragma: no cover
        self.RKN = False
        self.radius=False
        self.plot_stability(self.SDC, title="SDC stability region")

    def run_Picard_stability(self):  # pragma: no cover
        self.RKN = False
        self.radius=False
        self.plot_stability(self.picard, title="Picard stability region")

    def run_Ksdc(self):  # pragma: no cover
        self.radius=True
        self.plot_stability(self.Ksdc, title="$K_{sdc}$ spectral radius")

    def run_Kpicard(self):  # pragma: no cover
        self.radius=True
        self.plot_stability(self.Kpicard, title="$K_{picard}$ spectral radius")

    def run_RKN_stability(self):  # pragma: no cover
        self.RKN = True
        self.radius=False
        region_RKN = self.stability_data_RKN()
        self.plot_stability(region_RKN.T, title='RKN-4 stability region')
