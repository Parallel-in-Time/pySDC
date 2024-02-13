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


def plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, **kwargs):
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
    plot_options["figure_title"] = f"{label_from_key('dt')} VS {label_from_key('rel_err')}"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = True
    plot_options["log_base"] = 10
    plot_options["set_x_ticks_labels"] = True
    plot_options["set_y_ticks_labels"] = False
    # plot_options["xticks"]=[]
    # plot_options["yticks"]=[]
    plot_options["set_x_ticks_formatter"] = '{x:.1f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    if "slopes" in kwargs:
        plot_options["slopes"] = kwargs["slopes"]
    else:
        plot_options["slopes"] = [4]
    # plot_options["max_y"] = 50
    axes_data = [["level_params", 'dt'], ["perf_data", 'rel_err']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def cuboid_1D_one_level_diff_iter():
    experiment_name = "convergence"
    domain_name = "cuboid_1D"
    ionic_model_name = "BS"
    max_dt = 1.0
    min_dt_pow = 2
    max_dt_pow = 10
    n_time_ranks_list = [1, 1, 1, 1]

    num_nodes_list = [[4]]
    pre_refinements_list = [[2]]
    max_iter_list = [4]
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)


def cube_2D_one_level_diff_iter():
    experiment_name = "convergence_order_4"
    domain_name = "cube_2D"
    ionic_model_name = "TTP"
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 5
    n_time_ranks_list = [1, 1, 1, 1]

    num_nodes_list = [[4], [4], [4], [4]]
    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    max_iter_list = [1, 2, 3, 4]
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)

    num_nodes_list = [[6], [6], [6]]
    pre_refinements_list = [[-1], [-1], [-1]]
    max_iter_list = [2, 4, 6]
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)


def cube_2D_two_level_diff_iter():
    experiment_name = "convergence_order_4"
    domain_name = "cube_2D"
    ionic_model_name = "TTP"
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 3
    n_time_ranks_list = [1, 1]

    # num_nodes_list = [[4, 2], [4, 2]]
    # pre_refinements_list = [[-1, -2], [-1, -2]]
    # max_iter_list = [1, 2]
    # plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt)

    num_nodes_list = [[6, 3], [6, 3], [6, 3]]
    pre_refinements_list = [[-1, -2], [-1, -2], [-1, -2]]
    max_iter_list = [1, 2, 3]
    n_time_ranks_list = [1, 1, 1]
    slopes = [2, 4, 6]
    plot(experiment_name, domain_name, ionic_model_name, num_nodes_list, pre_refinements_list, max_iter_list, n_time_ranks_list, min_dt_pow, max_dt_pow, max_dt, slopes=slopes)


if __name__ == "__main__":
    cuboid_1D_one_level_diff_iter()
    # cuboid_1D_two_level_diff_iter()
    # cube_2D_one_level_diff_iter()
    # cube_2D_two_level_diff_iter()
