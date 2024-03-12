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


def files_list_dt_and_ranks(num_nodes, dt_list, pre_refinements, max_iter, n_time_ranks_list):
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
        for dt, n_time_ranks in zip(dt_list, n_time_ranks_list)
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


def read_results_parallel(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt):
    all_res = list()

    dt_list = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])
    for num_nodes, pre_refinements, max_iter, n_time_ranks in zip(num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list):
        files = files_list(num_nodes, dt_list, pre_refinements, max_iter, n_time_ranks)
        results = read_files_list(results_root, files)
        results["plot_label"] = label_from_data(num_nodes=None, pre_refinements=None, max_iter=None, n_time_ranks=n_time_ranks)
        all_res.append(results)

    return all_res


def read_results_parallel_dt_inv_prop_tasks(results_root, num_nodes_list, pre_refinements_list, max_iter_list, tend, min_dt_pow, max_dt_pow, max_dt):
    all_res = list()

    dt_list = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])
    n_time_ranks_list = [int(np.round(tend / dt)) for dt in dt_list]
    for num_nodes, pre_refinements, max_iter in zip(num_nodes_list, pre_refinements_list, max_iter_list):
        files = files_list_dt_and_ranks(num_nodes, dt_list, pre_refinements, max_iter, n_time_ranks_list)
        results = read_files_list(results_root, files)
        results["plot_label"] = label_from_data(num_nodes=None, pre_refinements=None, max_iter=None, n_time_ranks=None)
        all_res.append(results)

    return all_res


def plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside, **kwargs):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_res = read_results(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)

    for i, results in enumerate(all_res):
        ax2 = []
        for key, result in results.items():
            if key == "plot_label":
                continue
            ax2.append(result["perf_data"]["rel_err"])
        print(f"errs = {ax2}")

    output_file = output_folder / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_dt_VS_err.pdf")
    plot_options = dict()
    plot_options["location"] = "lower right"
    plot_options["figure_title"] = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = True
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = False
    plot_options["set_y_ticks_labels"] = False
    plot_options["set_x_label"] = True
    plot_options["set_y_label"] = True
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["slopes"] = slopes
    plot_options["legend_outside"] = legend_outside
    plot_options["show_legend"] = True
    # plot_options["max_y"] = 50
    axes_data = [["level_params", 'dt'], ["perf_data", 'rel_err']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def plot_parallel(
    experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside, **kwargs
):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_res = read_results_parallel(results_root, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)

    for i, results in enumerate(all_res):
        ax2 = []
        for key, result in results.items():
            if key == "plot_label":
                continue
            ax2.append(result["perf_data"]["rel_err"])
        print(f"errs = {ax2}")

    output_file = output_folder / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_dt_VS_err.pdf")
    plot_options = dict()
    plot_options["location"] = "upper left"
    plot_options["figure_title"] = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = True
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = False
    plot_options["set_y_ticks_labels"] = False
    plot_options["set_x_label"] = True
    plot_options["set_y_label"] = True
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["slopes"] = slopes
    plot_options["legend_outside"] = legend_outside
    plot_options["show_legend"] = True
    # plot_options["max_y"] = 50
    axes_data = [["level_params", 'dt'], ["perf_data", 'rel_err']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def plot_parallel_dt_inv_prop_tasks(
    experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, tend, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside, **kwargs
):
    save_plots_to_disk = True
    show_plots = True

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_res = read_results_parallel_dt_inv_prop_tasks(results_root, num_nodes_list, pre_refinements_list, max_iter_list, tend, min_dt_pow, max_dt_pow, max_dt)

    for i, results in enumerate(all_res):
        ax2 = []
        results["plot_label"] = results["plot_label"] + f"$P=T/\Delta t$"
        for key, result in results.items():
            if key == "plot_label":
                continue
            ax2.append(result["perf_data"]["rel_err"])
        print(f"errs = {ax2}")

    output_file = output_folder / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_dt_VS_err_dt_inv_prop_tasks.pdf")
    plot_options = dict()
    plot_options["location"] = "upper left"
    plot_options["figure_title"] = ""  # f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = True
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = False
    plot_options["set_y_ticks_labels"] = False
    plot_options["set_x_label"] = True
    plot_options["set_y_label"] = True
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    # plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["slopes"] = slopes
    plot_options["legend_outside"] = legend_outside
    plot_options["show_legend"] = True
    # plot_options["max_y"] = 50
    axes_data = [["level_params", 'dt'], ["perf_data", 'rel_err']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def fast_LA():
    experiment_name = "convergence_tend_1"
    domain_name = "03_fastl_LA"
    ionic_model_name = "TTP_SMOOTH"
    max_dt = 1.0
    min_dt_pow = 2
    max_dt_pow = 7
    n_time_ranks_list = [1, 1, 1, 1]

    num_nodes_list = [[6], [6], [6]]
    pre_refinements_list = [[0], [0], [0]]
    max_iter_list = [2, 4, 6]
    slopes = [[2, 4, 6], [0, 1, 2], [0.1, 0.1, 3.0], [[1, 5], [1, 5], [0, 3]]]
    legend_outside = True
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    num_nodes_list = [[6, 3], [6, 3], [6, 3]]
    pre_refinements_list = [[0], [0], [0]]
    max_iter_list = [1, 2, 3]
    slopes = [[2, 4, 6], [0, 1, 2], [8.0, 2.0, 0.2], [[2, 5], [1, 5], [1, 4]]]
    legend_outside = True
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)


def cube_1D():
    experiment_name = "convergence_parallel"
    domain_name = "cube_1D"
    ionic_model_name = "TTP_SMOOTH"
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 7

    # num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
    # n_time_ranks_list = [1, 4, 16, 64]
    # pre_refinements_list = [[0], [0], [0], [0]]
    # max_iter_list = [100, 100, 100, 100]
    # slopes = [[8], [0], [0.8], [[0, 4]]]
    # legend_outside = False
    # plot_parallel(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    num_nodes_list = [[6, 3]]
    tend = 16.0
    pre_refinements_list = [[0]]
    max_iter_list = [100]
    slopes = [[6], [0], [0.25], [[0, 6]]]
    legend_outside = False
    plot_parallel_dt_inv_prop_tasks(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, tend, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)


def cube_2D_ref__1():
    experiment_name = "convergence_tend_1"
    domain_name = "cube_2D"
    ionic_model_name = "TTP_SMOOTH"
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 6

    legend_outside = False

    n_time_ranks_list = [1, 1, 1, 1]
    # num_nodes_list = [[6], [6], [6]]
    # pre_refinements_list = [[-1], [-1], [-1]]
    # max_iter_list = [2, 4, 6]
    # slopes = [[2, 4, 6], [0, 1, 2], [0.1, 0.1, 0.1], [[3, 6], [3, 6], [3, 6]]]  # the slope and to which data it applies and a factor to translate it

    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    # num_nodes_list = [[6, 3], [6, 3], [6, 3]]
    # pre_refinements_list = [[-1], [-1], [-1]]
    # max_iter_list = [1, 2, 3]
    # slopes = [2, 4, 6]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes)

    # num_nodes_list = [[6, 3], [6, 3], [6, 3]]
    # pre_refinements_list = [[-1, -2], [-1, -2], [-1, -2]]
    # max_iter_list = [1, 2, 3]
    # slopes = [2, 4, 6]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes)

    # num_nodes_list = [[6, 3, 1], [6, 3, 1], [6, 3, 1]]
    # pre_refinements_list = [[-1], [-1]]
    # max_iter_list = [1, 2]
    # slopes = [[3, 5], [0, 1], [0.2, 0.05], [[3, 6], [3, 6]]]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    # num_nodes_list = [[4], [4], [4], [4]]
    # pre_refinements_list = [[-1], [-1], [-1], [-1]]
    # max_iter_list = [1, 2, 3, 4]
    # slopes = [[2, 4], [1, 3], [0.2, 0.15], [[3, 6], [3, 6], [3, 6], [3, 6]]]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    # num_nodes_list = [[4, 2], [4, 2]]
    # pre_refinements_list = [[-1], [-1]]
    # max_iter_list = [1, 2]
    # slopes = [[2, 4], [0, 1], [0.25, 0.15], [[3, 6], [3, 6]]]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    # n_time_ranks_list = [4, 4, 4]
    # num_nodes_list = [[4, 2], [4, 2], [4, 2]]
    # pre_refinements_list = [[-1], [-1], [-1]]
    # max_iter_list = [4, 6, 100]
    # slopes = [[4], [0], [0.25], [[3, 6]]]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    # num_nodes_list = [[4, 2], [4, 2]]
    # pre_refinements_list = [[-1, -2], [-1, -2]]
    # max_iter_list = [1, 2]
    # slopes = [[2, 4], [0, 1], [0.2, 0.15]]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes)

    num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
    n_time_ranks_list = [1, 4, 16, 64]
    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    max_iter_list = [100, 100, 100, 100]
    slopes = [[6], [0], [0.8], [[0, 4]]]
    legend_outside = False
    plot_parallel(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)

    num_nodes_list = [[6, 3]]
    tend = 16.0
    pre_refinements_list = [[-1]]
    max_iter_list = [100]
    slopes = [[6], [0], [0.25], [[0, 6]]]
    legend_outside = False
    plot_parallel_dt_inv_prop_tasks(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, tend, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)


def cube_2D_ref_0():
    experiment_name = "convergence_tend_1"
    domain_name = "cube_2D"
    ionic_model_name = "TTP_SMOOTH"
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 6
    n_time_ranks_list = [1, 1, 1, 1]

    num_nodes_list = [[6]]
    pre_refinements_list = [[0]]
    max_iter_list = [6]
    slopes = [[6], [0], [0.1], [[0, 6]]]  # the slope and to which data it applies and a factor to translate it
    legend_outside = False
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes, legend_outside)


if __name__ == "__main__":
    # cube_1D()
    cube_2D_ref__1()
    # cube_2D_ref_0()
    # cube_2D_ref__1_tmp()
    # fast_LA()
