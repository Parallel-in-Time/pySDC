import os
import numpy as np
from pathlib import Path
from pySDC.projects.Monodomain.plot_scripts.common import plot_results, read_files_list, label_from_data, pre_refinements_str, num_nodes_str, get_folders, label_from_key


def files_list(num_nodes, dt_list, pre_refinements, max_iter, n_time_ranks):
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
        for dt in dt_list
    ]


def read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt):
    all_res = list()

    dt_list = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])
    for num_nodes, pre_refinements, max_iter, n_time_ranks in zip(num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list):
        files = files_list(num_nodes, dt_list, pre_refinements, max_iter, n_time_ranks)
        results = read_files_list(results_root, files)
        results["plot_label"] = label_from_data(num_nodes=None, pre_refinements=None, max_iter=max_iter)
        all_res.append(results)

    return all_res


def files_list_nodes(num_nodes_list, dt, pre_refinements, max_iter, n_time_ranks):
    return [
        "pre_refinements_"
        + pre_refinements_str(pre_refinements)
        + "_num_nodes_"
        + str(num_nodes)
        + "_max_iter_"
        + str(max_iter)
        + "_dt_"
        + str(dt).replace(".", "p")
        + "_n_time_ranks_"
        + str(n_time_ranks)
        for num_nodes in num_nodes_list
    ]


def read_results_nodes(results_root, min_nodes, max_nodes, pre_refinements_list, max_iter_list, n_time_ranks_list, dt_list):
    all_res = list()

    num_nodes_list = list(range(min_nodes, max_nodes + 1))
    for dt, pre_refinements, max_iter, n_time_ranks in zip(dt_list, pre_refinements_list, max_iter_list, n_time_ranks_list):
        files = files_list_nodes(num_nodes_list, dt, pre_refinements, max_iter, n_time_ranks)
        results = read_files_list(results_root, files)
        results["plot_label"] = label_from_data(num_nodes=None, pre_refinements=None, max_iter=max_iter)
        all_res.append(results)

    return all_res


def plot(experiment_name, domain_name, ionic_model_names, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, fig_size):
    save_plots_to_disk = True
    show_plots = True

    all_res = dict()
    for ionic_model_name in ionic_model_names:
        results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
        all_res[ionic_model_name] = read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)

    for key, res in all_res.items():
        res[0]["plot_label"] = key

    all_res = [res[0] for key, res in all_res.items()]

    for i, results in enumerate(all_res):
        ax2 = []
        for key, result in results.items():
            if key == "plot_label":
                continue
            ax2.append(result["perf_data"]["mean_niters"])
        print(f"iter = {ax2}")

    output_file = output_folder.parent / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_dt_VS_iter.pdf")
    plot_options = dict()
    plot_options["location"] = "upper left"
    plot_options["figure_title"] = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = False
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = False
    plot_options["set_y_ticks_labels"] = False
    plot_options["set_x_label"] = True
    plot_options["set_y_label"] = True
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["legend_outside"] = False
    plot_options["max_y"] = 15
    plot_options["min_y"] = 0
    plot_options["fig_size"] = fig_size
    axes_data = [["level_params", 'dt'], ["perf_data", 'mean_niters'], ["perf_data", 'std_niters']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def plot_nodes(experiment_name, domain_name, ionic_model_names, pre_refinements_list, max_iter_list, n_time_ranks_list, min_nodes, max_nodes, dt_list):
    save_plots_to_disk = True
    show_plots = True

    all_res = dict()
    for ionic_model_name in ionic_model_names:
        results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
        all_res[ionic_model_name] = read_results_nodes(results_root, min_nodes, max_nodes, pre_refinements_list, max_iter_list, n_time_ranks_list, dt_list)

    for key, res in all_res.items():
        res[0]["plot_label"] = key

    all_res = [res[0] for key, res in all_res.items()]

    for i, results in enumerate(all_res):
        ax2 = []
        for key, result in results.items():
            if key == "plot_label":
                continue
            ax2.append(result["perf_data"]["mean_niters"])
        print(f"iter = {ax2}")

    output_file = output_folder.parent / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_dt_" + str(dt_list[0]).replace(".", "p") + "_nodes_VS_iter.pdf")
    plot_options = dict()
    plot_options["location"] = "upper right"
    plot_options["figure_title"] = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = False
    plot_options["logy"] = False
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = False
    plot_options["set_y_ticks_labels"] = False
    plot_options["set_x_label"] = True
    plot_options["set_y_label"] = True
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["legend_outside"] = False
    plot_options["max_y"] = 20
    axes_data = [["sweeper_params", 'num_nodes'], ["perf_data", 'mean_niters'], ["perf_data", 'std_niters']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def cube_2D_ref_0_dt():
    experiment_name = "iterations"
    domain_name = "cube_2D"
    max_dt = 1.0
    min_dt_pow = 2
    max_dt_pow = 7
    n_time_ranks_list = [1]
    ionic_model_names = ["HH", "CRN", "TTP"]
    pre_refinements_list = [[0]]
    max_iter_list = [100]

    num_nodes_list = [[8]]
    fig_size = (3, 2)
    plot(experiment_name, domain_name, ionic_model_names, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, fig_size)

    num_nodes_list = [[8, 6]]
    fig_size = (2, 2)
    plot(experiment_name, domain_name, ionic_model_names, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, fig_size)

    num_nodes_list = [[8, 4]]
    plot(experiment_name, domain_name, ionic_model_names, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, fig_size)

    num_nodes_list = [[8, 2]]
    plot(experiment_name, domain_name, ionic_model_names, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, fig_size)


def cube_2D_ref_0_nodes():
    experiment_name = "iterations"
    domain_name = "cube_2D"
    dt_list = [0.05]
    min_nodes = 1
    max_nodes = 12
    n_time_ranks_list = [1]
    ionic_model_names = ["HH", "CRN", "TTP"]
    pre_refinements_list = [[0]]
    max_iter_list = [100]

    plot_nodes(experiment_name, domain_name, ionic_model_names, pre_refinements_list, max_iter_list, n_time_ranks_list, min_nodes, max_nodes, dt_list)


if __name__ == "__main__":
    cube_2D_ref_0_dt()
    cube_2D_ref_0_nodes()
