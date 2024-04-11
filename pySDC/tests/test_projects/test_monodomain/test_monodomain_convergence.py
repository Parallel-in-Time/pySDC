import pytest


def run_monodomain_convergence(
    dt_max, n_dt, expected_convergence_rate, convergence_rate_tolerance, compute_init_val, compute_ref_sol, **opts
):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    opts["num_sweeps"] = [1]

    dt_list = [dt_max / 2**i for i in range(n_dt)]

    # skip residual computation at coarser levels (if any)
    opts["skip_residual_computation"] = True

    # interpolate or recompute rhs on fine level
    opts["finter"] = False

    # set time parallelism to True or emulated (False)
    opts["truly_time_parallel"] = False

    # set monodomain parameters
    opts["domain_name"] = "cuboid_1D_small"  # small problem for this pytest
    opts["refinements"] = [0]
    opts["order"] = 2  # 2 or 4
    opts["ionic_model_name"] = (
        "TTP_SMOOTH"  # a smoothed ionic model, the original TTP model has (very small) discontinuities due if-else statements in its implementation
    )
    opts["enable_output"] = False
    opts["write_database"] = False

    opts["output_root"] = "results_convergence"

    # save some values for later
    opts_bak = opts.copy()

    # In order to initiate an action potential the monodomain problem needs a stimulus. In our code the stimulus is a step function.
    # Due to its non smoothness we dont want to use it in the convergence test. Therefore we first generate an initial value,
    # using the step function, and then we use this initial value as the initial value for the convergence test. In that way the non smooth
    # stimulus is not used in the convergence test.

    # First, compute an initial value for the convergence test.
    opts["dt"] = 0.1
    opts["restol"] = 5e-8  # residual tolerance, doesn't need to be very small for the initial value
    opts["read_init_val"] = False
    opts["init_time"] = 0.0
    opts["end_time"] = 3.0
    opts["write_as_reference_solution"] = True  # write the initial value
    opts["write_all_variables"] = True  # write all variables, not only the potential
    opts["output_file_name"] = "init_val_DCT"
    opts["ref_sol"] = ""
    if compute_init_val:
        print("Computing initial value for the convergence test...")
        err, rel_err, avg_niters, times, niters, residuals = setup_and_run(**opts)

    # Second, compute a reference solution for the convergence test.
    opts["dt"] = dt_list[-1] / 4.0
    opts["restol"] = 1e-14  # residual tolerance, very small to no pollute convergence
    opts["read_init_val"] = True
    opts["init_time"] = 3.0  # start at t0=3
    opts["end_time"] = opts_bak["end_time"]  # end at t = t0+end_time
    opts["write_as_reference_solution"] = True  # write as reference solution
    opts["write_all_variables"] = (
        False  # write only the potential. The other ionic model variables are not taken in account in the convergence test.
    )
    opts["output_file_name"] = "ref_sol"
    if compute_ref_sol:
        print("Computing reference solution for the convergence test...")
        err, rel_err, avg_niters, times, niters, residuals = setup_and_run(**opts)

    # Third, run the convergence test
    opts["write_as_reference_solution"] = False
    opts["write_all_variables"] = False
    opts["ref_sol"] = "ref_sol"

    print("Running convergence test...")
    rel_err = [0.0] * n_dt
    for i, dt in enumerate(dt_list):
        print(f"Iteration {i} of {n_dt}...")
        opts["dt"] = dt
        opts["output_file_name"] = "monodomain_dt_" + str(dt).replace(".", "p")
        err, rel_err[i], avg_niters, times, niters, residuals = setup_and_run(**opts)

    import numpy as np

    rates = np.zeros(n_dt - 1)
    for i in range(n_dt - 1):
        rates[i] = np.log(rel_err[i] / rel_err[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])

    print("\nConvergence test results")
    print(f"Relative errors: {rel_err}")
    print(f"Rates: {rates}")

    assert np.all(rates > expected_convergence_rate - convergence_rate_tolerance), "ERROR: convergence rate is too low!"

    return dt_list, rel_err


@pytest.mark.monodomain
def test_monodomain_convergence_ESDC_TTP():
    max_iter_6_dt, max_iter_6_rel_err = run_monodomain_convergence(
        dt_max=0.2,
        n_dt=5,
        expected_convergence_rate=6.0,
        convergence_rate_tolerance=1.0,
        compute_init_val=True,
        compute_ref_sol=True,
        integrator="IMEXEXP_EXPRK",
        num_nodes=[6],
        max_iter=6,
        n_time_ranks=1,
        end_time=0.2,
    )

    max_iter_3_dt, max_iter_3_rel_err = run_monodomain_convergence(
        dt_max=0.2,
        n_dt=5,
        expected_convergence_rate=3.0,
        convergence_rate_tolerance=0.5,
        compute_init_val=False,
        compute_ref_sol=False,
        integrator="IMEXEXP_EXPRK",
        num_nodes=[6],
        max_iter=3,
        n_time_ranks=1,
        end_time=0.2,
    )

    import numpy as np

    max_iter_3_dt = np.array(max_iter_3_dt)
    max_iter_3_rel_err = np.array(max_iter_3_rel_err)
    max_iter_6_dt = np.array(max_iter_6_dt)
    max_iter_6_rel_err = np.array(max_iter_6_rel_err)

    import pySDC.helpers.plot_helper as plt_helper

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=0.89)

    lw = 1.5
    colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "x", "s", "D", "^"]

    plt_helper.plt.loglog(
        max_iter_3_dt,
        max_iter_3_rel_err,
        label="$k=3$",
        lw=lw,
        linestyle="-",
        color=colors[0],
        marker=markers[0],
        markerfacecolor="none",
        markeredgewidth=1.2,
        markersize=7.5,
    )
    plt_helper.plt.loglog(
        max_iter_6_dt,
        max_iter_6_rel_err,
        label="$k=6$",
        lw=lw,
        linestyle="-",
        color=colors[1],
        marker=markers[1],
        markerfacecolor="none",
        markeredgewidth=1.2,
        markersize=7.5,
    )
    plt_helper.plt.loglog(
        max_iter_3_dt,
        0.1 * np.min(max_iter_3_rel_err) * (max_iter_3_dt / max_iter_3_dt[-1]) ** 3,
        linewidth=2,
        linestyle="--",
        color="k",
        label=r"$\mathcal{{O}}(\Delta t^3)$",
    )
    plt_helper.plt.loglog(
        max_iter_6_dt,
        0.1 * np.min(max_iter_6_rel_err) * (max_iter_6_dt / max_iter_6_dt[-1]) ** 6,
        linewidth=2,
        linestyle="-",
        color="k",
        label=r"$\mathcal{{O}}(\Delta t^6)$",
    )
    plt_helper.plt.legend(loc="lower right", ncol=1)
    plt_helper.plt.ylabel('rel. err.')
    plt_helper.plt.xlabel(r"$\Delta t$")
    plt_helper.plt.grid()
    plt_helper.savefig("data/convergence_ESDC_fixed_iter", save_pdf=False, save_pgf=False, save_png=True)


if __name__ == "__main__":
    test_monodomain_convergence_ESDC_TTP()
