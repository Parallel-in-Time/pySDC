import os
import pickle

import pySDC.helpers.plot_helper as plt_helper


def plot_convergence():  # pragma: no cover
    '''
    Loads pickled error data for multiple preconditioners and collocation node counts and plots it with respect to the max. iteration count
    The y axis is logarithmically scaled. The x axis is linearly scaled.
    A new plot is generated for each preconditioner. Different collocation node counts are plotted on the same axes
    '''

    data = pickle.load(open("data/dae_conv_data.p", "rb"))

    # Configure specific line and symbol style_lists
    # These must match the data being loaded
    # General style_list settings e.g. font, should be changed in pySDC.helpers.plot_helper
    num_nodes_list = [3, 4, 5]
    color_list = ["r", "blue", "g"]
    shape_list = ["o", "d", "s"]
    start = 0
    end = 35
    plt_helper.setup_mpl()

    for qd_type in data.keys():
        fig, ax1 = plt_helper.newfig(textwidth=500, scale=0.89)  # Create a figure containing a single axes.
        ax2 = ax1.twinx()
        lns1 = list()
        lns2 = list()

        for num_nodes, color, shape in zip(num_nodes_list, color_list, shape_list):
            # Plot convergence data
            lns1.append(
                ax1.semilogy(
                    data[qd_type][num_nodes]['niter'][start:end],
                    data[qd_type][num_nodes]['error'][start:end],
                    label="Error {} nodes".format(num_nodes),
                    color=color,
                    marker=shape,
                    # ls=':',
                    lw=1,
                    alpha=0.4,
                )[0]
            )
            lns2.append(
                ax2.semilogy(
                    data[qd_type][num_nodes]['niter'][start:end],
                    data[qd_type][num_nodes]['residual'][start:end],
                    label="Residual {} nodes".format(num_nodes),
                    color=color,
                    marker=shape,
                    ls=':',
                    lw=1,
                    # alpha=0.4
                )[0]
            )

        ax1.set(xlabel='Iter. count', ylabel=r'$||u_1-\tilde{u}_1||_\infty$')
        ax1.grid(visible=True)
        ax2.set(ylabel=r'$||F\left(\tilde{u}, \tilde{u}\', t\right)||_\infty$')
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        # plt_helper.plt.show()
        fname = 'data/simple_dae_SDC_' + qd_type
        plt_helper.savefig(fname)

        assert os.path.isfile(fname + '.png')


if __name__ == "__main__":
    plot_convergence()
