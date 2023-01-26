import os
import sys
import numpy as np
import pickle

import pySDC.helpers.plot_helper as plt_helper


def plot_convergence():  # pragma: no cover
    '''
    Loads pickled error data for multiple preconditioners and collocation node count and plots it with respect to the time step size on a log-log axis.
    A new plot is generated for each preconditioner. Different collocation node counts are plotted on the same axes
    '''

    data = pickle.load(open("data/dae_conv_data.p", "rb"))

    # Configure specific line and symbol style_lists
    # These must match the data being loaded
    # General style_list settings e.g. font, should be changed in pySDC.helpers.plot_helper
    # num_nodes_list = [3, 4, 5]
    # color_list = ["r", "blue", "g"]
    # shape_list = ["o", "d", "s"]
    # style_list = [":", "-.", "--"]
    # order_list = [5, 7, 9]

    num_nodes_list = [3]
    color_list = ["blue"]
    shape_list = ["d"]
    style_list = ["-."]
    order_list = [5]

    # Some nasty hacking to get the iteration numbers positioned correctly individually in each plot
    num_data_points = len(data[next(iter(data))][num_nodes_list[0]]['error'])
    # adjust positions
    data['LU'][3]['position'] = ['center'] * num_data_points
    data['LU'][4]['position'] = ['center'] * num_data_points
    data['LU'][5]['position'] = ['center'] * num_data_points

    data['IE'][3]['position'] = ['center'] * num_data_points
    data['IE'][4]['position'] = ['center'] * num_data_points
    data['IE'][5]['position'] = ['center'] * num_data_points

    # data['MIN'][3]['position'] = ['center']*num_data_points
    # data['MIN'][4]['position'] = ['center']*num_data_points
    # data['MIN'][5]['position'] = ['center']*num_data_points

    # adjust offsets
    data['LU'][3]['offset'] = [(0, 10)] * num_data_points
    data['LU'][4]['offset'] = [(0, -10)] * 4
    data['LU'][4]['offset'].extend([(0, 10)] * (num_data_points - 4))
    data['LU'][5]['offset'] = [(0, -14)] * num_data_points

    data['IE'][3]['offset'] = [(0, 10)] * num_data_points
    data['IE'][4]['offset'] = [(0, -10)] * 4
    data['IE'][4]['offset'].extend([(0, 10)] * (num_data_points - 4))
    data['IE'][5]['offset'] = [(0, -14)] * num_data_points

    # data['MIN'][3]['offset'] = [(0, 10)]*num_data_points
    # data['MIN'][4]['offset'] = [(0, -10)]*4
    # data['MIN'][4]['offset'].extend([(0, 10)]*(num_data_points-4))
    # data['MIN'][5]['offset'] = [(0, -14)]*num_data_points

    plt_helper.setup_mpl()

    for qd_type in data.keys():
        fig, ax = plt_helper.newfig(textwidth=500, scale=0.89)  # Create a figure containing a single axes.
        # Init ylim to span largest possible interval. Refined in each iteration to fit data
        ylim = (sys.float_info.max, sys.float_info.min)

        for num_nodes, color, shape, style, order in zip(
            num_nodes_list, color_list, shape_list, style_list, order_list
        ):
            # Plot convergence data
            ax.loglog(
                data[qd_type][num_nodes]['dt'][data_start_point:],
                data[qd_type][num_nodes]['error'][data_start_point:],
                label="node count = {}".format(num_nodes),
                color=color,
                marker=shape,
                # ls=':',
                lw=1,
                # alpha=0.4
            )
            # Plot reference lines
            start_point = 3
            ax.loglog(
                data[qd_type][num_nodes]['dt'][start_point:],
                data[qd_type][num_nodes]['error'][start_point]
                * (data[qd_type][num_nodes]['dt'][start_point:] / data[qd_type][num_nodes]['dt'][start_point]) ** order,
                color="black",
                ls=style,
                lw=0.7,
                label="{}. order ref.".format(order),
            )
            # Write iteration count to each data point
            for niter, error, dt, position, offset in zip(
                data[qd_type][num_nodes]['niter'],
                data[qd_type][num_nodes]['error'],
                data[qd_type][num_nodes]['dt'],
                data[qd_type][num_nodes]['position'],
                data[qd_type][num_nodes]['offset'],
            ):
                ax.annotate(
                    niter,
                    (dt, error),
                    textcoords="offset points",  # how to position the text
                    xytext=offset,  # distance from text to points (x,y)
                    ha=position,
                )
            # Update the current y limits of the data
            # Ensures that final plot fits the data but cuts off the excess reference lines
            ylim = (
                min(np.append(data[qd_type][num_nodes]['error'][data_start_point:], ylim[0])),
                max(np.append(data[qd_type][num_nodes]['error'][data_start_point:], ylim[1])),
            )

        ax.set(ylim=((1e-2 * ylim[0], 5e1 * ylim[1])))
        ax.set(xlabel=r'$dt$', ylabel=r'$||u_5-\tilde{u}_5||_\infty$')
        ax.grid(visible=True)

        # reorder legend entries to place reference lines at end
        handles, labels = ax.get_legend_handles_labels()
        legend_order = range(len(handles))
        legend_order = np.concatenate(
            (list(filter(lambda x: x % 2 == 0, legend_order)), list(filter(lambda x: x % 2 == 1, legend_order)))
        )
        ax.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order])
        # plt_helper.plt.show()
        fname = 'data/simple_dae_SDC_' + qd_type
        plt_helper.savefig(fname)

        assert os.path.isfile(fname + '.png')


if __name__ == "__main__":
    plot_convergence()
