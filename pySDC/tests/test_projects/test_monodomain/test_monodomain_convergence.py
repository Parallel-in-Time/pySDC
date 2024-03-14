import pytest


def run_monodomain_convergence(
    integrator,
    num_nodes,
    max_iter,
    dt_max,
    n_dt,
    n_time_ranks,
    end_time_conv,
    expected_convergence_rate,
    convergence_rate_tolerance,
    compute_init_val,
    compute_ref_sol,
):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    num_sweeps = [1]

    dt_list = [dt_max / 2**i for i in range(n_dt)]

    # skip residual computation at coarser levels (if any)
    skip_residual_computation = True

    # interpolate or recompute rhs on fine level
    finter = False

    # set time parallelism to True or emulated (False)
    truly_time_parallel = False

    # set monodomain parameters
    domain_name = "cuboid_1D_small"  # small problem for this pytest
    refinements = [0]
    order = 2  # 2 or 4
    ionic_model_name = "TTP_SMOOTH"  # a smoothed ionic model, the original TTP model has (very small) discontinuities due if-else statements in its implementation
    enable_output = False
    write_database = False

    output_root = "results_convergence"

    # In order to initiate an action potential the monodomain problem needs a stimulus. In our code the stimulus is a step function.
    # Due to its non smoothness we dont want to use it in the convergence test. Therefore we first generate an initial value,
    # using the step function, and then we use this initial value as the initial value for the convergence test. In that way the non smooth
    # stimulus is not used in the convergence test.

    if compute_init_val:
        # First, compute an initial value for the convergence test.
        dt = 0.1
        restol = 5e-8  # residual tolerance, doesn't need to be very small for the initial value
        read_init_val = False
        init_time = 0.0
        end_time = 3.0
        write_as_reference_solution = True  # write the initial value
        write_all_variables = True  # write all variables, not only the potential
        output_file_name = "init_val_DCT"
        ref_sol = ""
        print("Computing initial value for the convergence test...")
        err, rel_err, avg_niters, times, niters, residuals = setup_and_run(
            integrator,
            num_nodes[:1],
            skip_residual_computation,
            num_sweeps,
            max_iter,
            dt,
            restol,
            domain_name,
            refinements,
            order,
            ionic_model_name,
            read_init_val,
            init_time,
            enable_output,
            write_as_reference_solution,
            write_all_variables,
            output_root,
            output_file_name,
            ref_sol,
            end_time,
            truly_time_parallel,
            1,
            finter,
            write_database,
        )

    if compute_ref_sol:
        # Second, compute a reference solution for the convergence test.
        dt = dt_list[-1] / 4.0
        restol = 1e-14  # residual tolerance, very small to no pollute convergence
        read_init_val = True
        init_time = 3.0  # start at t0=3
        end_time = end_time_conv  # end at t = t0+end_time
        write_as_reference_solution = True  # write refernece solution
        write_all_variables = False  # write only the potential. The other ionic model variables are not taken in account in the convergence test.
        output_file_name = "ref_sol"
        ref_sol = ""
        print("Computing reference solution for the convergence test...")
        err, rel_err, avg_niters, times, niters, residuals = setup_and_run(
            integrator,
            num_nodes[:1],
            skip_residual_computation,
            num_sweeps,
            max_iter,
            dt,
            restol,
            domain_name,
            refinements,
            order,
            ionic_model_name,
            read_init_val,
            init_time,
            enable_output,
            write_as_reference_solution,
            write_all_variables,
            output_root,
            output_file_name,
            ref_sol,
            end_time,
            truly_time_parallel,
            1,
            finter,
            write_database,
        )

    # Third, run the convergence test
    restol = 1e-14  # residual tolerance, very small to no pollute convergence
    read_init_val = True
    init_time = 3.0  # start at t0=3
    end_time = end_time_conv  # end at t = t0+end_time
    write_as_reference_solution = False
    write_all_variables = False
    ref_sol = "ref_sol"

    print("Running convergence test...")
    rel_err = [0.0] * n_dt
    for i, dt in enumerate(dt_list):
        print(f"Iteration {i} of {n_dt}...")
        output_file_name = "monodomain_dt_" + str(dt).replace(".", "p")
        err, rel_err[i], avg_niters, times, niters, residuals = setup_and_run(
            integrator,
            num_nodes,
            skip_residual_computation,
            num_sweeps,
            max_iter,
            dt,
            restol,
            domain_name,
            refinements,
            order,
            ionic_model_name,
            read_init_val,
            init_time,
            enable_output,
            write_as_reference_solution,
            write_all_variables,
            output_root,
            output_file_name,
            ref_sol,
            end_time,
            truly_time_parallel,
            n_time_ranks,
            finter,
            write_database,
        )

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
        integrator="IMEXEXP_EXPRK",
        num_nodes=[6],
        max_iter=6,
        dt_max=0.2,
        n_dt=5,
        n_time_ranks=1,
        end_time_conv=0.2,
        expected_convergence_rate=6.0,
        convergence_rate_tolerance=1.0,
        compute_init_val=True,
        compute_ref_sol=True,
    )

    max_iter_3_dt, max_iter_3_rel_err = run_monodomain_convergence(
        integrator="IMEXEXP_EXPRK",
        num_nodes=[6],
        max_iter=3,
        dt_max=0.2,
        n_dt=5,
        n_time_ranks=1,
        end_time_conv=0.2,
        expected_convergence_rate=3.0,
        convergence_rate_tolerance=0.5,
        compute_init_val=False,
        compute_ref_sol=False,
    )

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rc("text", usetex=True)
    font = {'family': 'serif', 'serif': ['computer modern roman']}
    plt.rc('font', **font)

    max_iter_3_dt = np.array(max_iter_3_dt)
    max_iter_3_rel_err = np.array(max_iter_3_rel_err)
    max_iter_6_dt = np.array(max_iter_6_dt)
    max_iter_6_rel_err = np.array(max_iter_6_rel_err)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.plot(
        max_iter_3_dt,
        max_iter_3_rel_err,
        label="$k=3$",
        linewidth=2,
        marker="o",
        color="C0",
        markerfacecolor="none",
        markeredgewidth=1.2,
        markersize=7.5,
    )
    ax.plot(
        max_iter_6_dt,
        max_iter_6_rel_err,
        label="$k=6$",
        linewidth=2,
        marker="x",
        color="C1",
        markerfacecolor="none",
        markeredgewidth=1.2,
        markersize=7.5,
    )
    ax.plot(
        max_iter_3_dt,
        0.1 * np.min(max_iter_3_rel_err) * (max_iter_3_dt / max_iter_3_dt[-1]) ** 3,
        linewidth=2,
        linestyle="--",
        color="k",
        label="$\mathcal{{O}}(\Delta t^3)$",
    )
    ax.plot(
        max_iter_6_dt,
        0.1 * np.min(max_iter_6_rel_err) * (max_iter_6_dt / max_iter_6_dt[-1]) ** 6,
        linewidth=2,
        linestyle="-",
        color="k",
        label="$\mathcal{{O}}(\Delta t^6)$",
    )
    ax.set_xlabel("$\Delta t$", fontsize=12)
    ax.set_ylabel("rel. err.", fontsize=12)
    ax.set_title("Convergence for fixed iterations")
    ax.legend(loc="lower right", facecolor='white', framealpha=0.95)
    # # plt.show()
    # executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    # fig.savefig(
    #     executed_file_dir + "/../../../../data/convergence_ESDC_fixed_iter.png", bbox_inches="tight", format="png"
    # )


if __name__ == "__main__":
    test_monodomain_convergence_ESDC_TTP()
