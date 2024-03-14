import pytest


@pytest.mark.monodomain
def test_monodomain_convergence_one_level():
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    integrator = "IMEXEXP_EXPRK"
    num_nodes = [4]
    num_sweeps = [1]

    # set step parameters
    max_iter = 4

    # set level parameters
    dt_max = 0.1
    n_dt = 4
    dt_list = [dt_max / 2**i for i in range(n_dt)]

    # skip residual computation at coarser levels (if any)
    skip_residual_computation = True

    # interpolate or recompute rhs on fine level
    finter = False

    # set time parallelism to True or emulated (False)
    truly_time_parallel = False
    n_time_ranks = 1

    # set monodomain parameters
    domain_name = "cuboid_1D_small"  # small problem for this pytest
    refinements = [0]
    order = 2  # 2 or 4
    ionic_model_name = "TTP_SMOOTH"  # a smoothed ionic model, the original TTP model has (very small) discontinuities due if-else statements in its implementation
    enable_output = False

    output_root = "results_pytest"

    # In order to initiate an action potential the monodomain problem needs a stimulus. In our code the stimulus is a step function.
    # Due to its non smoothness we dont want to use it in the convergence test. Therefore we first generate an initial value,
    # using the step function, and then we use this initial value as the initial value for the convergence test. In that way the non smooth
    # stimulus is not used in the convergence test.

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
    err, rel_err, avg_niters = setup_and_run(
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
    )

    # Second, compute a reference solution for the convergence test.
    dt = dt_list[-1] / 4.0
    restol = 1e-14  # residual tolerance, very small to no pollute convergence
    read_init_val = True
    init_time = 3.0  # start at t0=3
    end_time = 1.0  # end at t = t0+end_time = 4
    write_as_reference_solution = True  # write refernece solution
    write_all_variables = False  # write only the potential. The other ionic model variables are not taken in account in the convergence test.
    output_file_name = "ref_sol"
    ref_sol = ""

    print("Computing reference solution for the convergence test...")
    err, rel_err, avg_niters = setup_and_run(
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
    )

    # Third, run the convergence test
    read_init_val = True
    init_time = 3.0  # start at t0=3
    end_time = 1.0  # end at t = t0+end_time = 4
    write_as_reference_solution = False
    write_all_variables = False
    ref_sol = "ref_sol"

    print("Running convergence test...")
    rel_err = [0.0] * n_dt
    for i, dt in enumerate(dt_list):
        print(f"Iteration {i} of {n_dt}...")
        output_file_name = "monodomain_dt_" + str(dt).replace(".", "p")
        err, rel_err[i], avg_niters = setup_and_run(
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
        )

    import numpy as np

    rates = np.zeros(n_dt - 1)
    for i in range(n_dt - 1):
        rates[i] = np.log(rel_err[i] / rel_err[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])

    print("\nConvergence test results")
    print(f"Relative errors: {rel_err}")
    print(f"Rates: {rates}")

    assert np.all(rates > 3.5), "ERROR: convergence rate is too low!"
    assert np.all(rates < 4.5), "ERROR: convergence rate is too high!"


@pytest.mark.monodomain
def test_monodomain_convergence_two_levels():
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    integrator = "IMEXEXP_EXPRK"
    num_nodes = [4, 2]
    num_sweeps = [1]

    # set step parameters
    max_iter = 3

    # set level parameters
    dt_max = 0.05
    n_dt = 4
    dt_list = [dt_max / 2**i for i in range(n_dt)]

    # skip residual computation at coarser levels (if any)
    skip_residual_computation = True

    # interpolate or recompute rhs on fine level
    finter = False

    # set time parallelism to True or emulated (False)
    truly_time_parallel = False
    n_time_ranks = 1

    # set monodomain parameters
    domain_name = "cuboid_1D_small"
    refinements = [0]
    order = 2  # 2 or 4
    ionic_model_name = "TTP_SMOOTH"  # a smoothed ionic model, the original TTP model has (very small) discontinuities due if-else statements in its implementation
    enable_output = False

    output_root = "results_convergence_pytest"

    # In order to initiate an action potential the monodomain problem needs a stimulus. In our code the stimulus is a step function.
    # Due to its non smoothness we dont want to use it in the convergence test. Therefore we first generate an initial value,
    # using the step function, and then we use this initial value as the initial value for the convergence test. In that way the non smooth
    # stimulus is not used in the convergence test.

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
    err, rel_err, avg_niters = setup_and_run(
        integrator,
        num_nodes[:1],
        skip_residual_computation,
        num_sweeps,
        2 * max_iter,
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
    )

    # Second, compute a reference solution for the convergence test.
    dt = dt_list[-1] / 4.0
    restol = 1e-14  # residual tolerance, very small to no pollute convergence
    read_init_val = True
    init_time = 3.0  # start at t0=3
    end_time = 1.0  # end at t = t0+end_time
    write_as_reference_solution = True  # write refernece solution
    write_all_variables = False  # write only the potential. The other ionic model variables are not taken in account in the convergence test.
    output_file_name = "ref_sol"
    ref_sol = ""

    print("Computing reference solution for the convergence test...")
    err, rel_err, avg_niters = setup_and_run(
        integrator,
        num_nodes[:1],
        skip_residual_computation,
        num_sweeps,
        2 * max_iter,
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
    )

    # Third, run the convergence test
    read_init_val = True
    init_time = 3.0  # start at t0=3
    end_time = 1.0  # end at t = t0+end_time
    write_as_reference_solution = False
    write_all_variables = False
    ref_sol = "ref_sol"

    print("Running convergence test...")
    rel_err = [0.0] * n_dt
    for i, dt in enumerate(dt_list):
        print(f"Iteration {i} of {n_dt}...")
        output_file_name = "monodomain_dt_" + str(dt).replace(".", "p")
        err, rel_err[i], avg_niters = setup_and_run(
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
        )

    import numpy as np

    rates = np.zeros(n_dt - 1)
    for i in range(n_dt - 1):
        rates[i] = np.log(rel_err[i] / rel_err[i + 1]) / np.log(dt_list[i] / dt_list[i + 1])

    print("\nConvergence test results")
    print(f"Relative errors: {rel_err}")
    print(f"Rates: {rates}")

    assert np.all(rates > 3.5), "ERROR: convergence rate is too low!"
    assert np.all(rates < 4.5), "ERROR: convergence rate is too high!"


# if __name__ == "__main__":
# test_monodomain_convergence_one_level()
# test_monodomain_convergence_two_levels()
