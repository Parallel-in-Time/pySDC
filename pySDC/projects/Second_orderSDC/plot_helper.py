# import matplotlib

# matplotlib.use('Agg')
# import os

import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 16
FIG_SIZE = (7.44, 6.74)


def set_fixed_plot_params():  # pragma: no cover
    """
    Set fixed parameters for all plots
    """
    plt.rcParams['figure.figsize'] = FIG_SIZE
    plt.rcParams['pgf.rcfonts'] = False

    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = FONT_SIZE + 5
    plt.rcParams['axes.labelsize'] = FONT_SIZE + 5
    plt.rcParams['xtick.labelsize'] = FONT_SIZE
    plt.rcParams['ytick.labelsize'] = FONT_SIZE
    plt.rcParams['xtick.major.pad'] = 5
    plt.rcParams['ytick.major.pad'] = 5
    plt.rcParams['axes.labelpad'] = 6
    plt.rcParams['lines.markersize'] = FONT_SIZE - 2
    plt.rcParams['lines.markeredgewidth'] = 1
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rc('font', size=FONT_SIZE)


class PlotManager(object):  # pragma: no cover
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

    def plot_convergence(self):
        """
        Plot convergence order plots for the position and velocity
        If you change parameters of the values you need set y_lim values need to set manually
        """
        set_fixed_plot_params()
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
                    size=18,
                )
            else:
                ax1.text(
                    time_data[ii, 1],
                    0.25 * convline['pos'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii, 0, 0]),
                    size=18,
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
                    size=18,
                )
            else:
                ax2.text(
                    time_data[ii, 1],
                    0.25 * convline['vel'][value, ii, 1],
                    r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0, 0]),
                    size=18,
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
        if self.error_type == 'global':
            ax2.set_ylim(1e-14, 1e1)
            ax1.set_ylim(1e-14, 1e1)
        else:
            ax2.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
            ax1.set_ylim(np.min(ax1.get_ylim()), np.max(ax2.get_ylim()))
        ax2.legend(loc='best')
        fig2.tight_layout()
        plt.show()
        fig2.savefig(self.cwd + 'data/{}_conv_plot_vel{}.pdf'.format(self.error_type, value + 1))

    def format_number(self, data_value, indx):
        """
        Change format of the x axis for the work precision plots
        """
        if data_value >= 1_000_000:
            formatter = "{:1.1f}M".format(data_value * 0.000_001)
        else:
            formatter = "{:1.0f}K".format(data_value * 0.001)
        return formatter

    def plot_work_precision(self):
        """
        Generate work precision plots
        """
        set_fixed_plot_params()
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
                label='Velocity-Verlet',
            )
            ax2.loglog(
                func_eval_VV[0],
                error_VV['vel'][value,][0][:],
                ls='dashdot',
                color='blue',
                marker='H',
                label='Velocity-Verlet',
            )

        for ii, jj in enumerate(self.K_iter):
            # =============================================================================
            # # If you want to get exactly the same picture like in paper uncomment this only for vertical axis
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
                label='k={}'.format(jj),
            )
            ax1.loglog(
                func_eval_picard[ii, :], error_picard['pos'][value, ii, :], ls='--', color=color[ii], marker=shape[ii]
            )

            ax2.loglog(
                func_eval_SDC[ii, :],
                error_SDC['vel'][value, ii, :],
                ls='solid',
                color=color[ii],
                marker=shape[ii],
                label='k={}'.format(jj),
            )
            ax2.loglog(
                func_eval_picard[ii, :], error_picard['vel'][value, ii, :], ls='--', color=color[ii], marker=shape[ii]
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

        ax1.legend(loc="best", fontsize=12)
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

    def organize_data(self, filename='data/dt_vs_local_errorSDC.csv', time_iter=None):
        """
        Organize data according to plot
        Args:
            filename (string): data to find approximate order
            time_iter : in case it you used different time iterations
        """
        if time_iter == None:
            time_iter = self.time_iter

        items = np.genfromtxt(filename, delimiter=',', skip_header=1)
        time = items[:, 0]
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
            error_data['pos'][ii, :, :] = items[:, ii + 3].reshape([N, time_iter])
            error_data['vel'][ii, :, :] = items[:, ii + 8].reshape([N, time_iter])

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
                file = open(self.cwd + 'data/{}_order_vs_approx_order.csv'.format(self.error_type), 'w')

            else:
                file = open(self.cwd + 'data/{}_order_vs_approx_order.csv'.format(self.error_type), 'a')

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
