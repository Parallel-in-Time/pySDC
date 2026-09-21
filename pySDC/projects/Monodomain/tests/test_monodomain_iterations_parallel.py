import pytest


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

    for i, (iters_info, label) in enumerate(zip(iters_info_list, labels_list, strict=True)):
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


def _data_file(options):
    """Where `setup_and_run` puts the database for a given set of options."""
    from pathlib import Path

    return (
        Path(__file__).resolve().parents[3]
        / "data"
        / options["output_root"]
        / options["domain_name"]
        / f"ref_{options['refinements'][0]}"
        / options["ionic_model_name"]
        / options["output_file_name"]
    )


def run_and_check_iterations(expected_avg_niters, **options):
    """
    Run one configuration and check the iteration count it reports.

    This used to serialise `options` into a command line, launch `run_MonodomainODE_cli.py` under
    `mpirun`, and then recover the iteration count by reading the database the run had written --
    while `setup_and_run` returns it directly. Both round trips are gone; `generate_initial_value`
    above was already calling the same function this way.
    """
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    options["num_sweeps"] = [1]
    options["max_iter"] = 100
    options["dt"] = 0.025
    options["restol"] = 5e-8
    options["end_time"] = 0.6
    options["domain_name"] = "cuboid_1D_small"
    options["refinements"] = [0]
    options["order"] = 4
    options["read_init_val"] = True
    options["init_time"] = 3.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["write_all_variables"] = False
    options["output_root"] = "results_iterations_parallel"
    # `skip_res` was the command line's name for it
    options["skip_residual_computation"] = True
    options["finter"] = False
    options["write_database"] = True
    # never set by this test: the command line was supplying its own default
    options["ref_sol"] = "ref_sol"

    _, _, avg_niters, times, niters, residuals = setup_and_run(**options)

    from mpi4py import MPI

    if MPI.COMM_WORLD.rank == 0:
        print(f"Got average number of iterations {avg_niters}, expected was {expected_avg_niters}")
        assert avg_niters == pytest.approx(
            expected_avg_niters, rel=0.1
        ), f"Average number of iterations {avg_niters} too different from the expected {expected_avg_niters}"

    return {"avg_niters": avg_niters, "times": times, "niters": niters, "residuals": residuals}


# Each configuration writes under its own name, so the three databases coexist for the plot below.
CASES = {
    "ESDC": dict(num_nodes=[8], n_time_ranks=1, expected_avg_niters=3.58333),
    "MLESDC": dict(num_nodes=[8, 4], n_time_ranks=1, expected_avg_niters=2.0),
    "PFASST": dict(num_nodes=[8, 4], n_time_ranks=24, expected_avg_niters=3.0),
}


def _run_case(name):
    case = dict(CASES[name])
    return run_and_check_iterations(
        integrator="IMEXEXP_EXPRK",
        ionic_model_name="TTP",
        truly_time_parallel=True,
        output_file_name=f"monodomain_{name}",
        **case,
    )


@pytest.fixture(scope="module", autouse=True)
def initial_value():
    """
    Every run below reads its initial value from file, so it has to be written first.

    Once per session, not once per test -- and the 24-rank job is a session of its own, so it
    cannot rely on the serial one having produced it. Rank 0 writes while the others wait, since
    this is a serial computation and they would otherwise race on the same file.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        generate_initial_value(ionic_model_name="TTP")
    comm.Barrier()


# PFASST first, so a plain `pytest` run (where mpi-pytest forks per test) produces its data before
# the plot below. Under `etc/run_mpi_tests.sh` the rank passes run before the serial pass anyway.
@pytest.mark.monodomain
@pytest.mark.parallel(24)
def test_monodomain_iterations_PFASST():
    _run_case("PFASST")


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC():
    _run_case("ESDC")


@pytest.mark.monodomain
def test_monodomain_iterations_MLESDC():
    _run_case("MLESDC")


@pytest.mark.monodomain
def test_plot_iterations():
    """
    Draw the figures the website shows, from the databases the three runs above wrote.

    They cannot be passed in memory: PFASST runs in a 24-rank job of its own, so the database is
    how its result reaches this process. That is the one round trip through storage that is
    actually load-bearing.
    """
    from pySDC.projects.Monodomain.utils.data_management import database

    iters_info_list = []
    for name in CASES:
        options = dict(
            CASES[name],
            output_root="results_iterations_parallel",
            domain_name="cuboid_1D_small",
            refinements=[0],
            ionic_model_name="TTP",
            output_file_name=f"monodomain_{name}",
        )
        iters_info_list.append(database(str(_data_file(options))).read_dictionary("iters_info"))

    for key2, logy, ylabel, title, out in [
        ("niters", False, r"\# iter", "Number of iterations", "niter_VS_time"),
        ("residuals", True, "residual", "Residual over time", "res_VS_time"),
    ]:
        plot_iter_info(
            iters_info_list,
            list(CASES),
            key1="times",
            key2=key2,
            logy=logy,
            xlabel="$t$",
            ylabel=ylabel,
            ymin=None,
            ymax=None,
            title=title,
            output_file_name=out,
        )
