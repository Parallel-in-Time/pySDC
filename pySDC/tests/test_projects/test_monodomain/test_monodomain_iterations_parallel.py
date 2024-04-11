import pytest
import os
import subprocess


def plot_iter_info(iters_info_list, labels_list, key1, key2, logy, xlabel, ylabel, ymin, ymax, title, output_file_name):

    markers = ["o", "x", "s", "D", "v", "^", "<", ">", "p", "h", "H", "*", "+", "X", "d", "|", "_"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    import pySDC.helpers.plot_helper as plt_helper

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=0.89)

    lw = 1.5
    colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "x", "s", "D", "^"]

    if logy:
        plt_helper.plt.yscale("log", base=10)

    for i, (iters_info, label) in enumerate(zip(iters_info_list, labels_list)):
        plt_helper.plt.plot(
            iters_info[key1],
            iters_info[key2],
            label=label,
            lw=lw,
            color=colors[i],
            marker=markers[i],
            markerfacecolor="none",
            markeredgewidth=1.2,
            markersize=7.5,
        )

    if ymin is not None and ymax is not None:
        plt_helper.plt.set_ylim([ymin, ymax])

    plt_helper.plt.legend(loc="lower right", ncol=1)
    plt_helper.plt.ylabel(ylabel)
    plt_helper.plt.xlabel(xlabel)
    plt_helper.plt.title(title)
    plt_helper.plt.grid()
    plt_helper.savefig("data/" + output_file_name, save_pdf=False, save_pgf=False, save_png=True)


def options_command(options):
    cmd = ""
    for key, val in options.items():
        if type(val) is list:
            opt = key
            if type(val[0]) is int:
                arg = ",".join([str(v).replace("-", "_") for v in val])
            else:
                arg = ",".join([map(str, val)])
        elif type(val) is bool:
            if not val:
                opt = "no-" + key
            else:
                opt = key
            arg = ""
        else:
            opt = key
            arg = str(val)
        cmd = cmd + " --" + opt + (" " + arg if arg != "" else "")
    return cmd


def generate_initial_value(ionic_model_name):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    opts = dict()

    # define sweeper parameters
    opts["integrator"] = "IMEXEXP_EXPRK"
    opts["num_nodes"] = [5]
    opts["num_sweeps"] = [1]

    # set step parameters
    opts["max_iter"] = 100

    # set level parameters
    opts["dt"] = 0.1

    opts["restol"] = 5e-8  # residual tolerance

    opts["truly_time_parallel"] = False
    opts["n_time_ranks"] = 1

    # skip residual computation at coarser levels (if any)
    opts["skip_residual_computation"] = True

    # interpolate or recompute rhs on fine level
    opts["finter"] = False

    # set monodomain parameters
    opts["domain_name"] = "cuboid_1D_small"
    opts["ionic_model_name"] = ionic_model_name
    opts["refinements"] = [0]
    opts["order"] = 4  # 2 or 4

    opts["enable_output"] = False
    opts["write_database"] = False

    opts["output_root"] = "results_iterations_parallel"

    opts["read_init_val"] = False
    opts["init_time"] = 0.0
    opts["end_time"] = 6.0
    opts["write_as_reference_solution"] = True
    opts["write_all_variables"] = True
    opts["output_file_name"] = "init_val_DCT"
    opts["ref_sol"] = ""

    err, rel_err, avg_niters, times, niters, residuals = setup_and_run(**opts)


def check_iterations_parallel(expected_avg_niters, **options):
    # define sweeper parameters

    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100
    options["dt"] = 0.025

    # set level parameters
    options["restol"] = 5e-8

    options["end_time"] = 0.6

    # set problem parameters
    options["domain_name"] = "cuboid_1D_small"
    options["refinements"] = [0]
    options["order"] = 4
    options["read_init_val"] = True
    options["init_time"] = 3.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["write_all_variables"] = False
    options["output_file_name"] = "monodomain"
    options["output_root"] = "results_iterations_parallel"
    options["skip_res"] = True
    options["finter"] = False
    options["write_database"] = True

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '.:../../../..'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = "pySDC/projects/Monodomain/run_scripts"

    # base_python_command = "coverage run -p run_MonodomainODE_cli.py"
    base_python_command = "coverage run -p " + cwd + "/run_MonodomainODE_cli.py"
    cmd = f"mpirun -n {options['n_time_ranks']} " + base_python_command + " " + options_command(options)

    print(f"Running command: {cmd}")

    process = subprocess.Popen(
        args=cmd.split(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env,
        cwd=".",
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    process.wait()

    assert (
        process.returncode == 0
    ), f"ERROR: did not get return code 0, got {process.returncode} with {options['n_time_ranks']} processes"

    # read the generated data
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        executed_file_dir
        + "/../../../../data/"
        + options["output_root"]
        + "/"
        + options["domain_name"]
        + "/ref_"
        + str(options["refinements"][0])
        + "/"
        + options["ionic_model_name"]
        + "/"
        + options["output_file_name"]
    )
    from pySDC.projects.Monodomain.utils.data_management import database

    data_man = database(file_name)
    # errors = data_man.read_dictionary("errors")
    iters_info = data_man.read_dictionary("iters_info")

    print(f"Got average number of iterations {iters_info['avg_niters']}, expected was {expected_avg_niters}")

    assert iters_info['avg_niters'] == pytest.approx(
        expected_avg_niters, rel=0.1
    ), f"Average number of iterations {iters_info['avg_niters']} too different from the expected {expected_avg_niters}"

    return iters_info


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_MLESDC_PFASST():

    generate_initial_value(ionic_model_name="TTP")

    ESDC_iters_info = check_iterations_parallel(
        integrator="IMEXEXP_EXPRK",
        num_nodes=[8],
        ionic_model_name="TTP",
        truly_time_parallel=True,
        n_time_ranks=1,
        expected_avg_niters=3.58333,
    )

    MLESDC_iters_info = check_iterations_parallel(
        integrator="IMEXEXP_EXPRK",
        num_nodes=[8, 4],
        ionic_model_name="TTP",
        truly_time_parallel=True,
        n_time_ranks=1,
        expected_avg_niters=2.0,
    )

    PFASST_iters_info = check_iterations_parallel(
        integrator="IMEXEXP_EXPRK",
        num_nodes=[8, 4],
        ionic_model_name="TTP",
        truly_time_parallel=True,
        n_time_ranks=24,
        expected_avg_niters=3.0,
    )

    iters_info_list = [ESDC_iters_info, MLESDC_iters_info, PFASST_iters_info]
    labels_list = ["ESDC", "MLESDC", "PFASST"]
    plot_iter_info(
        iters_info_list,
        labels_list,
        key1='times',
        key2='niters',
        logy=False,
        xlabel="$t$",
        ylabel=r"\# iter",
        ymin=None,
        ymax=None,
        title="Number of iterations",
        output_file_name="niter_VS_time",
    )
    plot_iter_info(
        iters_info_list,
        labels_list,
        key1='times',
        key2='residuals',
        logy=True,
        xlabel="$t$",
        ylabel="residual",
        ymin=None,
        ymax=None,
        title="Residual over time",
        output_file_name="res_VS_time",
    )


if __name__ == "__main__":
    test_monodomain_iterations_ESDC_MLESDC_PFASST()
