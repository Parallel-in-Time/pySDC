import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pySDC.projects.Monodomain.plot_scripts.common import plot_results, read_files_list, label_from_data, pre_refinements_str, num_nodes_str, get_folders, label_from_key

import matplotlib.font_manager

matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)


def files_list(num_nodes, dt, pre_refinements, max_iter, n_time_ranks):
    return [
        "pre_refinements_"
        + pre_refinements_str(pre_refinements)
        + "_num_nodes_"
        + num_nodes_str(num_nodes)
        + "_max_iter_"
        + str(max_iter)
        + "_dt_"
        + str(dt).replace(".", "p")
        + "_n_time_ranks_"
        + str(n_time_ranks)
    ]


def read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt):
    all_res = list()

    for num_nodes, pre_refinements, max_iter, n_time_ranks in zip(num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list):
        files = files_list(num_nodes, dt, pre_refinements, max_iter, n_time_ranks)
        results = read_files_list(results_root, files)
        if results == {}:
            continue
        results = results[list(results.keys())[0]]
        results["plot_label"] = label_from_data(num_nodes=num_nodes, pre_refinements=None, max_iter=None, n_time_ranks=n_time_ranks)
        all_res.append(results)

    return all_res


def plot_last_res_vs_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_results = read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)

    output_file = output_folder.parent / Path("num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_time_VS_res.pdf")
    location = "upper left"
    figure_title = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    logx = False
    logy = True
    log_base = 10
    set_x_ticks_labels = False
    set_y_ticks_labels = False
    set_x_label = True
    set_y_label = True
    xticks = None
    yticks = None
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    set_x_ticks_formatter = None
    set_y_ticks_formatter = None
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    max_y = 1e-7
    min_y = 1e-9
    legend_outside = True
    show_legend = False
    markevery = 15
    axes_data = [["perf_data", 'times'], ["perf_data", 'last_residual']]

    set_x_ticks_formatter = None
    set_y_ticks_formatter = None

    export_legend = True
    from pySDC.projects.Monodomain.plot_scripts.common import get_plot_settings

    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize, lw, slopes_style = get_plot_settings()

    figsize = (7, 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log", base=log_base)
    if logy:
        ax.set_yscale("log", base=log_base)

    # all_res is a list of dictionaries, each dictionary will generate a curve in the plot
    # each dictionary key is a string naming an experiment, each value is a dictionary with some data about the results of this experiment

    ax1_data = axes_data[0]
    ax2_data = axes_data[1]

    ax1 = []
    ax2 = []
    min_ax2 = []
    for i, results in enumerate(all_results):
        ax1.append(results[ax1_data[0]][ax1_data[1]])
        ax2.append(results[ax2_data[0]][ax2_data[1]])
        label = results["plot_label"]
        ax1[-1] = list(np.array(ax1[-1]).flatten())
        if ax2[-1] != []:
            min_ax2.append(min(ax2[-1]))
        ax.plot(
            ax1[-1],
            ax2[-1],
            label=label,
            linewidth=lw,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
            markevery=markevery,
        )

    if set_x_ticks_formatter is not None:
        ax.xaxis.set_major_formatter(set_x_ticks_formatter)
    if set_y_ticks_formatter is not None:
        ax.yaxis.set_major_formatter(set_y_ticks_formatter)

    ax.xaxis.set_minor_formatter('')
    ax.yaxis.set_minor_formatter('')

    fs_label = 12
    fs_tick = 12
    if set_x_label:
        ax.set_xlabel(label_from_key(ax1_data[1]), fontsize=fs_label, labelpad=-0.5)
    if set_y_label:
        ax.set_ylabel(label_from_key(ax2_data[1]), fontsize=fs_label, labelpad=-0.5)
    ax.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax.set_ylim([min_y, max_y])

    def merge_ax(ax):
        merged_ax = ax[0]
        for i in range(1, len(ax)):
            for j in range(len(ax[i])):
                if not np.any(np.isclose(merged_ax, ax[i][j])):
                    merged_ax.append(ax[i][j])
        merged_ax.sort()
        return merged_ax

    merged_ax1 = merge_ax(ax1)
    merged_ax2 = merge_ax(ax2)

    if set_x_ticks_labels:
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(merged_ax1[: len(merged_ax1) : 2])
    if set_y_ticks_labels:
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(merged_ax2)

    if figure_title != "":
        ax.set_title(figure_title)

    if not legend_outside:
        ax.legend(loc=location, facecolor='white', framealpha=0.95)
    else:
        n = len(all_results)
        order = list(range(n))
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.4, 1.0), loc='lower center', ncols=n)
        # ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncols=n)

    if save_plots_to_disk and export_legend:
        legend_file_name = output_file.with_name(output_file.stem + "_legend.pdf")
        from pySDC.projects.Monodomain.plot_scripts.common import export_legend_image

        export_legend_image(ax.get_legend(), legend_file_name)

    if not show_legend:
        ax.get_legend().remove()

    if show_plots:
        plt.show()

    if save_plots_to_disk:
        if not output_file.parent.is_dir():
            os.makedirs(output_file.parent, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight", format="pdf")

    plt.close(fig)


def plot_niter_vs_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_results = read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)

    output_file = output_folder.parent / Path("num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_niter_VS_res.pdf")
    location = "upper left"
    figure_title = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    logx = False
    logy = False
    log_base = 10
    set_x_ticks_labels = False
    set_y_ticks_labels = False
    set_x_label = True
    set_y_label = True
    xticks = None
    yticks = None
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    set_x_ticks_formatter = None
    set_y_ticks_formatter = None
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    max_y = None
    min_y = None
    legend_outside = True
    show_legend = False
    markevery = 10
    axes_data = [["perf_data", 'times'], ["perf_data", 'niters']]

    set_x_ticks_formatter = None
    set_y_ticks_formatter = None

    export_legend = True
    from pySDC.projects.Monodomain.plot_scripts.common import get_plot_settings

    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize, lw, slopes_style = get_plot_settings()

    figsize = (7, 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log", base=log_base)
    if logy:
        ax.set_yscale("log", base=log_base)

    # all_res is a list of dictionaries, each dictionary will generate a curve in the plot
    # each dictionary key is a string naming an experiment, each value is a dictionary with some data about the results of this experiment

    ax1_data = axes_data[0]
    ax2_data = axes_data[1]

    ax1 = []
    ax2 = []
    min_ax2 = []
    for i, results in enumerate(all_results):
        ax1.append(results[ax1_data[0]][ax1_data[1]])
        ax2.append(results[ax2_data[0]][ax2_data[1]])
        label = results["plot_label"]
        ax1[-1] = list(np.array(ax1[-1]).flatten())
        if ax2[-1] != []:
            min_ax2.append(min(ax2[-1]))
        ax.plot(
            ax1[-1],
            ax2[-1],
            label=label,
            linewidth=lw,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
            markevery=markevery,
        )

    if set_x_ticks_formatter is not None:
        ax.xaxis.set_major_formatter(set_x_ticks_formatter)
    if set_y_ticks_formatter is not None:
        ax.yaxis.set_major_formatter(set_y_ticks_formatter)

    ax.xaxis.set_minor_formatter('')
    ax.yaxis.set_minor_formatter('')

    fs_label = 12
    fs_tick = 12
    if set_x_label:
        ax.set_xlabel(label_from_key(ax1_data[1]), fontsize=fs_label, labelpad=-0.5)
    if set_y_label:
        ax.set_ylabel(label_from_key(ax2_data[1]), fontsize=fs_label, labelpad=-0.5)
    ax.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax.set_ylim([min_y, max_y])

    def merge_ax(ax):
        merged_ax = ax[0]
        for i in range(1, len(ax)):
            for j in range(len(ax[i])):
                if not np.any(np.isclose(merged_ax, ax[i][j])):
                    merged_ax.append(ax[i][j])
        merged_ax.sort()
        return merged_ax

    merged_ax1 = merge_ax(ax1)
    merged_ax2 = merge_ax(ax2)

    if set_x_ticks_labels:
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(merged_ax1[: len(merged_ax1) : 2])
    if set_y_ticks_labels:
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(merged_ax2)

    if figure_title != "":
        ax.set_title(figure_title)

    if not legend_outside:
        ax.legend(loc=location, facecolor='white', framealpha=0.95)
    else:
        n = len(all_results)
        order = list(range(n))
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.4, 1.0), loc='lower center', ncols=n)
        # ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncols=n)

    if save_plots_to_disk and export_legend:
        legend_file_name = output_file.with_name(output_file.stem + "_legend.pdf")
        from pySDC.projects.Monodomain.plot_scripts.common import export_legend_image

        export_legend_image(ax.get_legend(), legend_file_name)

    if not show_legend:
        ax.get_legend().remove()

    if show_plots:
        plt.show()

    if save_plots_to_disk:
        if not output_file.parent.is_dir():
            os.makedirs(output_file.parent, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight", format="pdf")

    plt.close(fig)


def plot_res_ev_at_given_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt, time):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_results = read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)

    # for results in all_res:
    #     print(f"last_residual = {results['perf_data']['last_residual']}")

    output_file = output_folder.parent / Path("num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_time_" + str(time).replace(".", "p") + "_iter_VS_res.pdf")
    location = "upper left"
    figure_title = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    logx = False
    logy = True
    log_base = 10
    set_x_ticks_labels = False
    set_y_ticks_labels = False
    set_x_label = True
    set_y_label = True
    xticks = None
    yticks = None
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    set_x_ticks_formatter = None
    set_y_ticks_formatter = None
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    max_x = 7.5
    min_x = 0.5
    max_y = None
    min_y = None
    legend_outside = True
    show_legend = False
    markevery = 1

    set_x_ticks_formatter = None
    set_y_ticks_formatter = None

    export_legend = False
    from pySDC.projects.Monodomain.plot_scripts.common import get_plot_settings

    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize, lw, slopes_style = get_plot_settings()

    figsize = (3, 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log", base=log_base)
    if logy:
        ax.set_yscale("log", base=log_base)

    # all_res is a list of dictionaries, each dictionary will generate a curve in the plot
    # each dictionary key is a string naming an experiment, each value is a dictionary with some data about the results of this experiment

    ax1 = []
    ax2 = []
    min_ax2 = []
    for i, results in enumerate(all_results):
        times = results["perf_data"]["times"]
        index = np.argmin(np.abs(np.array(times) - time))
        ax1.append(list(range(1, int(results["perf_data"]["niters"][index]) + 1)))
        ax2.append(results["perf_data"]["residuals"][index])
        label = results["plot_label"]
        if ax2[-1] != []:
            min_ax2.append(min(ax2[-1]))
        ax.plot(
            ax1[-1],
            ax2[-1],
            label=label,
            linewidth=lw,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
            markevery=markevery,
        )

    if set_x_ticks_formatter is not None:
        ax.xaxis.set_major_formatter(set_x_ticks_formatter)
    if set_y_ticks_formatter is not None:
        ax.yaxis.set_major_formatter(set_y_ticks_formatter)

    ax.xaxis.set_minor_formatter('')
    ax.yaxis.set_minor_formatter('')

    fs_label = 12
    fs_tick = 12
    if set_x_label:
        ax.set_xlabel("$k$", fontsize=fs_label, labelpad=-0.5)
    if set_y_label:
        ax.set_ylabel("res", fontsize=fs_label, labelpad=-0.5)
    ax.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax.set_ylim([min_y, max_y])
    ax.set_xlim([min_x, max_x])

    def merge_ax(ax):
        merged_ax = ax[0]
        for i in range(1, len(ax)):
            for j in range(len(ax[i])):
                if not np.any(np.isclose(merged_ax, ax[i][j])):
                    merged_ax.append(ax[i][j])
        merged_ax.sort()
        return merged_ax

    merged_ax1 = merge_ax(ax1)
    merged_ax2 = merge_ax(ax2)

    if set_x_ticks_labels:
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(merged_ax1[: len(merged_ax1) : 2])
    if set_y_ticks_labels:
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(merged_ax2)

    if figure_title != "":
        ax.set_title(figure_title)

    if not legend_outside:
        ax.legend(loc=location, facecolor='white', framealpha=0.95)
    else:
        n = len(all_results)
        order = list(range(n))
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.4, 1.0), loc='lower center', ncols=n)
        # ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncols=n)

    if save_plots_to_disk and export_legend:
        legend_file_name = output_file.with_name(output_file.stem + "_legend.pdf")
        from pySDC.projects.Monodomain.plot_scripts.common import export_legend_image

        export_legend_image(ax.get_legend(), legend_file_name)

    if not show_legend:
        ax.get_legend().remove()

    if show_plots:
        plt.show()

    if save_plots_to_disk:
        if not output_file.parent.is_dir():
            os.makedirs(output_file.parent, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight", format="pdf")

    plt.close(fig)


def cube_1D():
    experiment_name = "residuals"
    domain_name = "cube_1D"
    dt = 0.05
    n_time_ranks_list = [1, 1, 8, 256]
    num_nodes_list = [[8]] + [[8, 4]] * len(n_time_ranks_list[1:])
    ionic_model_name = "TTP"
    pre_refinements_list = [[0], [0], [0], [0]]
    max_iter_list = [100, 100, 100, 100]

    plot_last_res_vs_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)

    time = dt * 256
    plot_res_ev_at_given_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt, time)

    time = dt * 128
    plot_res_ev_at_given_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt, time)


def cube_2D():
    experiment_name = "residuals"
    domain_name = "cube_2D"
    dt = 0.05
    n_time_ranks_list = [1, 1, 8, 256]
    num_nodes_list = [[8]] + [[8, 4]] * len(n_time_ranks_list[1:])
    ionic_model_name = "TTP"
    pre_refinements_list = [[0], [0], [0], [0]]
    max_iter_list = [100, 100, 100, 100]

    plot_last_res_vs_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)
    plot_niter_vs_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt)

    time = dt * 256
    plot_res_ev_at_given_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt, time)

    time = dt * 128
    plot_res_ev_at_given_time(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, dt, time)


if __name__ == "__main__":
    # cube_1D()
    cube_2D()
