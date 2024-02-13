import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pySDC.projects.Monodomain.utils.data_management import database

import os
from pathlib import Path
import matplotlib.font_manager

matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)


def read_file(file):
    if not file.with_suffix('.db').is_file():
        print(f"File {str(file)} does not exist")
        return None

    data_man = database(file)
    res = dict()
    res["problem_params"] = data_man.read_dictionary("problem_params")
    res["step_params"] = data_man.read_dictionary("step_params")
    res["sweeper_params"] = data_man.read_dictionary("sweeper_params")
    res["level_params"] = data_man.read_dictionary("level_params")
    res["space_transfer_params"] = data_man.read_dictionary("space_transfer_params")
    res["base_transfer_params"] = data_man.read_dictionary("base_transfer_params")
    res["controller_params"] = data_man.read_dictionary("controller_params")
    res["perf_data"] = data_man.read_dictionary("perf_data")
    return res


def get_plot_settings():
    markers = ["o", "x", "s", ">", ".", "<", ",", "1", "2", "3", "4", "v", "p", "*", "h", "H", "+", "^", "D", "d", "|", "_"]
    colors = [f"C{i}" for i in range(10)]
    # colors[1], colors[2], colors[3], colors[7] = colors[2], colors[3], colors[7], colors[1]
    markerfacecolors = ["none" for _ in range(len(colors))]
    # markerfacecolors[3] = colors[3]
    markersizes = [7.5 for _ in range(len(colors))]
    markeredgewidths = [1.2 for _ in range(len(colors))]
    figsize = (3, 2)
    lw = 2
    slopes_style = dict()
    slopes_style["linewidth"] = 2
    slopes_style["linestyle"] = ["--", "-.", "-"]
    slopes_style["color"] = ["k", "k", "k"]
    return markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize, lw, slopes_style


def plot_results(all_results, axes_data, save_plots_to_disk, show_plots, output_file, plot_options):
    location = plot_options["location"]
    figure_title = plot_options["figure_title"]
    with_legend = plot_options["with_legend"]
    logx = plot_options["logx"]
    logy = plot_options["logy"]
    log_base = plot_options["log_base"] if "log_base" in plot_options else 10
    max_y = plot_options["max_y"] if "max_y" in plot_options else None
    min_y = plot_options["min_y"] if "min_y" in plot_options else None
    slopes = plot_options["slopes"] if "slopes" in plot_options else None
    set_x_ticks_formatter = plot_options["set_x_ticks_formatter"] if "set_x_ticks_formatter" in plot_options else None
    set_y_ticks_formatter = plot_options["set_y_ticks_formatter"] if "set_y_ticks_formatter" in plot_options else None
    set_x_ticks_labels = plot_options["set_x_ticks_labels"] if "set_x_ticks_labels" in plot_options else False
    set_y_ticks_labels = plot_options["set_y_ticks_labels"] if "set_y_ticks_labels" in plot_options else False
    xticks = plot_options["xticks"] if "xticks" in plot_options else None
    yticks = plot_options["yticks"] if "yticks" in plot_options else None

    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize, lw, slopes_style = get_plot_settings()
    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log", base=log_base)
    if logy:
        ax.set_yscale("log", base=log_base)

    # all_res is a list of dictionaries, each dictionary will generate a curve in the plot
    # each dictionary key is a string naming an experiment, each value is a dictionary with some data about the results of this experiment

    if len(axes_data) == 3:
        with_std = True
    else:
        with_std = False

    ax1_data = axes_data[0]
    ax2_data = axes_data[1]

    min_max_ax2 = np.inf

    for i, results in enumerate(all_results):
        ax1 = []
        ax2 = []
        if with_std:
            ax3 = []
        label = results["plot_label"]
        del results["plot_label"]
        for key, result in results.items():
            ax1.append(result[ax1_data[0]][ax1_data[1]])
            ax2.append(result[ax2_data[0]][ax2_data[1]])
            if with_std:
                ax3.append(result[axes_data[2][0]][axes_data[2][1]])
        if ax2 != []:
            min_max_ax2 = min(min_max_ax2, max(ax2))
        ax.plot(
            ax1,
            ax2,
            label=label,
            linewidth=lw,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
        )
        if with_std:
            y1 = np.array(ax2) - np.array(ax3)
            y2 = np.array(ax2) + np.array(ax3)
            ax.fill_between(ax1, y1, y2, color=colors[i], alpha=0.1)

    if slopes is not None:
        ax1_np = np.array(ax1)
        for i, slope in enumerate(slopes):
            ax.plot(
                ax1_np,
                min_max_ax2 * (ax1_np / ax1_np[0]) ** slope,
                linewidth=slopes_style["linewidth"],
                linestyle=slopes_style["linestyle"][i],
                color=slopes_style["color"][i],
                label=f"$\mathcal{{O}}(\Delta t^{{{slope}}})$",
            )

    if set_x_ticks_formatter is not None:
        ax.xaxis.set_major_formatter(set_x_ticks_formatter)
    if set_y_ticks_formatter is not None:
        ax.yaxis.set_major_formatter(set_y_ticks_formatter)

    ax.xaxis.set_minor_formatter('')
    ax.yaxis.set_minor_formatter('')

    fs_label = 12
    fs_tick = 12
    ax.set_xlabel(label_from_key(ax1_data[1]), fontsize=fs_label, labelpad=-0.5)
    ax.set_ylabel(label_from_key(ax2_data[1]), fontsize=fs_label, labelpad=-0.5)
    ax.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax.set_ylim([min_y, max_y])

    if set_x_ticks_labels:
        ax1_np = np.array(ax1)
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(ax1_np[range(0, len(ax1), 2)])
    if set_y_ticks_labels:
        ax2_np = np.array(ax2)
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(ax2_np)

    if figure_title != "":
        ax.set_title(figure_title)

    if with_legend:
        ax.legend(loc=location)

    if show_plots:
        plt.show()

    if save_plots_to_disk:
        if not output_file.parent.is_dir():
            os.makedirs(output_file.parent, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight", format="pdf")
        # export_legend(ax.get_legend())

    plt.close(fig)


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def label_from_key(key):
    if key == "dt":
        return "$\Delta t$"
    elif key == "cpu_time":
        return "CPU time"
    elif key == "rel_err":
        return "$L^2$-norm rel. err."
    elif key == "n_time_ranks":
        return "\#ranks"
    elif key == "mean_niters":
        return "avg \#iterations"
    else:
        return key


def read_files_list(results_root, files):
    files_res = dict()
    for file in files:
        res = read_file(results_root / Path(file))
        if res is not None:
            files_res[file] = res
    return files_res


def num_nodes_str(num_nodes):
    return "-".join([str(node) for node in num_nodes])


def num_nodes_str_for_label(num_nodes):
    return "(" + ",".join([str(node) for node in num_nodes]) + ")"


def pre_refinements_str(pre_refinements):
    return "-".join([str(pre_refinement) for pre_refinement in pre_refinements])


def label_from_data(num_nodes=None, pre_refinements=None, max_iter=None, dt=None):
    label = []
    if num_nodes is not None:
        label.append("$m = " + num_nodes_str_for_label(num_nodes) + "$")
    if pre_refinements is not None:
        label.append("$p = " + pre_refinements_str(pre_refinements) + "$")
    if max_iter is not None:
        label.append("$k = " + str(max_iter) + "$")
    if dt is not None:
        label.append("$\Delta t = " + str(dt) + "$")
    return ",".join(label)


def get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name):
    subfolder = Path("results_" + experiment_name) / Path(domain_name) / Path("ref_" + str(pre_refinements_list[0][0])) / Path(ionic_model_name)

    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    results_root = executed_file_dir + "/../../../../data/Monodomain/"
    results_root = results_root / subfolder

    output_folder = Path.home() / Path("Dropbox/Ricerca/Articoli/TIME-X/Article 10 - PinT for Monodomain equation/images")
    output_folder = output_folder / subfolder

    return results_root, output_folder
