import os
import numpy as np
from pathlib import Path
from pySDC.projects.Monodomain.plot_scripts.common import plot_results, read_files_list, label_from_data, pre_refinements_str, num_nodes_str, get_folders, label_from_key


def files_list(num_nodes, dt, pre_refinements, n_time_ranks_list):
    return [
        "pre_refinements_" + pre_refinements_str(pre_refinements) + "_num_nodes_" + num_nodes_str(num_nodes) + "_dt_" + str(dt).replace(".", "p") + "_n_time_ranks_" + str(n_time_ranks)
        for n_time_ranks in n_time_ranks_list
    ]


def read_results(results_root, num_nodes_list, dt_list, pre_refinements_list, n_time_ranks_list):
    all_res = list()

    for num_nodes, dt, pre_refinements in zip(num_nodes_list, dt_list, pre_refinements_list):
        files = files_list(num_nodes, dt, pre_refinements, n_time_ranks_list)
        results = read_files_list(results_root, files)
        results["plot_label"] = label_from_data(num_nodes=None, pre_refinements=None, max_iter=None, dt=dt)
        all_res.append(results)

    return all_res


def plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list):
    save_plots_to_disk = True
    show_plots = False

    results_root, output_folder = get_folders(experiment_name, domain_name, pre_refinements_list, ionic_model_name)
    all_res = read_results(results_root, num_nodes_list, dt_list, pre_refinements_list, n_time_ranks_list)

    output_file = output_folder / Path("pre_refinements_" + pre_refinements_str(pre_refinements_list[0]) + "_num_nodes_" + num_nodes_str(num_nodes_list[0]) + "_n_time_ranks_VS_n_iters.pdf")
    plot_options = dict()
    plot_options["location"] = "upper left"
    plot_options["figure_title"] = "\#ranks VS avg \#iterations"
    plot_options["with_legend"] = True
    plot_options["logx"] = True
    plot_options["logy"] = False
    plot_options["log_base"] = 2
    plot_options["set_x_ticks_labels"] = True
    plot_options["set_y_ticks_labels"] = False
    plot_options["xticks"] = n_time_ranks_list[::2]
    # plot_options["yticks"]=[]
    plot_options["set_x_ticks_formatter"] = '{x:.0f}'
    # plot_options["set_y_ticks_formatter"] = '{x:.1f}'
    plot_options["min_y"] = 0
    plot_options["max_y"] = 40
    axes_data = [["perf_data", 'n_time_ranks'], ["perf_data", 'mean_niters'], ["perf_data", 'std_niters']]
    plot_results(all_res, axes_data, save_plots_to_disk, show_plots, output_file, plot_options)


def dependency_on_step_size_cuboid_1D_very_large():
    experiment_name = "stability"
    domain_name = "cuboid_1D_very_large"

    n_time_ranks_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dt_list = [0.025, 0.05, 0.1, 0.2]

    # ionic_model_name_list = ["BS", "HH", "CRN", "TTP"]
    ionic_model_name_list = ["TTP"]

    for ionic_model_name in ionic_model_name_list:
        pre_refinements_list = [[2], [2], [2], [2]]
        num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

        pre_refinements_list = [[2], [2], [2], [2]]
        num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

        pre_refinements_list = [[2, 1], [2, 1], [2, 1], [2, 1]]
        num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

        pre_refinements_list = [[2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0]]
        num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)


def dependency_on_step_size_cuboid_1D_very_large_DCT():
    experiment_name = "stability_DCT"
    domain_name = "cuboid_1D_very_large"

    n_time_ranks_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dt_list = [0.025, 0.05, 0.1, 0.2]

    ionic_model_name = "TTP"

    for i in [2, 3, 4, 5]:
        pre_refinements_list = [[i], [i], [i], [i]]
        num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

        pre_refinements_list = [[i, i - 1], [i, i - 1], [i, i - 1], [i, i - 1]]
        num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
        plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)


def dependency_on_step_size_cube_1D_DCT():
    experiment_name = "stability_DCT"
    domain_name = "cube_1D"

    n_time_ranks_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dt_list = [0.025, 0.05, 0.1, 0.2]

    ionic_model_name_list = ["HH", "TTP"]
    for ionic_model_name in ionic_model_name_list:
        for i in [2, 4]:
            pre_refinements_list = [[i], [i], [i], [i]]
            num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
            plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

            pre_refinements_list = [[i, i - 1], [i, i - 1], [i, i - 1], [i, i - 1]]
            num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
            plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

            pre_refinements_list = [[i], [i], [i], [i]]
            num_nodes_list = [[6, 3, 1], [6, 3, 1], [6, 3, 1], [6, 3, 1]]
            plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

            pre_refinements_list = [[i, i - 1, i - 2], [i, i - 1, i - 2], [i, i - 1, i - 2], [i, i - 1, i - 2]]
            num_nodes_list = [[6, 3, 1], [6, 3, 1], [6, 3, 1], [6, 3, 1]]
            plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)


def dependency_on_step_size_cube_2D():
    experiment_name = "stability"
    # domain_name = "cuboid_1D_very_large"
    # pre_refinements = [2]
    domain_name = "cube_2D"

    n_time_ranks_list = [1, 2, 4, 8, 16, 32, 64, 128]
    dt_list = [0.025, 0.05, 0.1, 0.2]

    ionic_model_name = "TTP"

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1, -2], [-1, -2], [-1, -2], [-1, -2]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1, -2, -3], [-1, -2, -3], [-1, -2, -3], [-1, -2, -3]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 3, 1], [6, 3, 1], [6, 3, 1], [6, 3, 1]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1, -2], [-1, -2], [-1, -2], [-1, -2]]
    num_nodes_list = [[6, 3], [6, 3], [6, 3], [6, 3]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    dt_list = [0.025, 0.05, 0.1]
    pre_refinements_list = [[-1, -2, -3], [-1, -2, -3], [-1, -2, -3], [-1, -2, -3]]
    num_nodes_list = [[6, 3, 1], [6, 3, 1], [6, 3, 1], [6, 3, 1]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    dt_list = [0.025, 0.1]
    pre_refinements_list = [[0], [0]]
    num_nodes_list = [[6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0, -1], [0, -1]]
    num_nodes_list = [[6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    dt_list = [0.025, 0.1]
    pre_refinements_list = [[0], [0]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0, -1, -2], [0, -1, -2]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    # --------------------------------------------------------------------------------------------------------------------
    ionic_model_name = "CRN"

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1], [-1], [-1], [-1]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1, -2], [-1, -2], [-1, -2], [-1, -2]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[-1, -2, -3], [-1, -2, -3], [-1, -2, -3], [-1, -2, -3]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0], [0], [0], [0]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0], [0], [0], [0]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0, -1], [0, -1], [0, -1], [0, -1]]
    num_nodes_list = [[6, 4], [6, 4], [6, 4], [6, 4]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)

    pre_refinements_list = [[0, -1, -2], [0, -1, -2], [0, -1, -2], [0, -1, -2]]
    num_nodes_list = [[6, 4, 2], [6, 4, 2], [6, 4, 2], [6, 4, 2]]
    plot(experiment_name, domain_name, ionic_model_name, pre_refinements_list, num_nodes_list, dt_list, n_time_ranks_list)


if __name__ == "__main__":
    dependency_on_step_size_cuboid_1D_very_large()
    # dependency_on_step_size_cube_2D()
    # dependency_on_step_size_cuboid_1D_very_large_DCT()
    # dependency_on_step_size_cube_1D_DCT()
