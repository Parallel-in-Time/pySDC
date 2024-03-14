import pytest


def generate_initial_value(integrator, num_nodes, ionic_model_name):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    num_sweeps = [1]

    # set step parameters
    max_iter = 100

    # set level parameters
    dt = 0.05

    restol = 5e-8  # residual tolerance

    truly_time_parallel = False
    n_time_ranks = 1

    # skip residual computation at coarser levels (if any)
    skip_residual_computation = True

    # interpolate or recompute rhs on fine level
    finter = False

    # set monodomain parameters
    domain_name = "cuboid_1D_small"
    refinements = [0]
    order = 4  # 2 or 4
    enable_output = False

    output_root = "results_iterations_parallel_pytest"

    read_init_val = False
    init_time = 0.0
    end_time = 3.0
    write_as_reference_solution = True
    write_all_variables = True
    output_file_name = "init_val_DCT"
    ref_sol = ""

    err, rel_err, avg_niters, iter_counts = setup_and_run(
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


def check_iterations_parallel(
    integrator, num_nodes, ionic_model_name, truly_time_parallel, n_time_ranks, expected_avg_niters
):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    num_sweeps = [1]

    # set step parameters
    max_iter = 100

    # set level parameters
    dt = 0.025

    restol = 5e-8  # residual tolerance

    # skip residual computation at coarser levels (if any)
    skip_residual_computation = True

    # interpolate or recompute rhs on fine level
    finter = False

    # set monodomain parameters
    domain_name = "cuboid_1D_small"
    refinements = [0]
    order = 4  # 2 or 4
    enable_output = False

    output_root = "results_iterations_parallel_pytest"

    read_init_val = True
    init_time = 3.0
    end_time = 3.2  # 128 steps of 0.025
    write_as_reference_solution = False
    write_all_variables = False
    output_file_name = "monodomain"
    ref_sol = ""

    err, rel_err, avg_niters, iter_counts = setup_and_run(
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

    print(f"Got average number of iterations {avg_niters}, expected was {expected_avg_niters}")

    # assert avg_niters == pytest.approx(
    #     expected_avg_niters, rel=0.1
    # ), f"Average number of iterations {avg_niters} too different from the expected {expected_avg_niters}"

    return iter_counts


if __name__ == "__main__":
    # generate_initial_value(integrator="IMEXEXP_EXPRK", num_nodes=[5], ionic_model_name="TTP")
    # ESDC_iter_counts = check_iterations_parallel(
    #     integrator="IMEXEXP_EXPRK",
    #     num_nodes=[6],
    #     ionic_model_name="TTP",
    #     truly_time_parallel=False,
    #     n_time_ranks=1,
    #     expected_avg_niters=3.7734375,
    # )
    # MLESDC_iter_counts = check_iterations_parallel(
    #     integrator="IMEXEXP_EXPRK",
    #     num_nodes=[6,3],
    #     ionic_model_name="TTP",
    #     truly_time_parallel=False,
    #     n_time_ranks=1,
    #     expected_avg_niters=2.890625,
    # )
    # PFASST_4_iter_counts = check_iterations_parallel(
    #     integrator="IMEXEXP_EXPRK",
    #     num_nodes=[6, 3],
    #     ionic_model_name="TTP",
    #     truly_time_parallel=False,
    #     n_time_ranks=4,
    #     expected_avg_niters=2.4921875,
    # )
    PFASST_16_iter_counts = check_iterations_parallel(
        integrator="IMEXEXP_EXPRK",
        num_nodes=[6, 3],
        ionic_model_name="TTP",
        truly_time_parallel=False,
        n_time_ranks=16,
        expected_avg_niters=3.8671875,
    )
