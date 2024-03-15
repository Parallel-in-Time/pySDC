import pytest


def check_iterations(num_nodes, ionic_model_name, expected_avg_niters):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    integrator = "IMEXEXP_EXPRK"
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

    # set time parallelism to True or emulated (False)
    truly_time_parallel = False
    n_time_ranks = 1

    # set monodomain parameters
    domain_name = "cuboid_2D_small"
    refinements = [-1]
    order = 4  # 2 or 4
    enable_output = False
    write_database = False

    output_root = "results_iterations_pytest"

    read_init_val = False
    init_time = 0.0
    end_time = 4.0
    write_as_reference_solution = False
    write_all_variables = False
    output_file_name = "monodomain"
    ref_sol = ""

    err, rel_err, avg_niters, times, niters, residuals = setup_and_run(
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

    print(f"Got average number of iterations {avg_niters}, expected was {expected_avg_niters}")

    assert avg_niters == pytest.approx(
        expected_avg_niters, rel=0.1
    ), f"Average number of iterations {avg_niters} too different from the expected {expected_avg_niters}"


# Many of the following are commented since they test features already tested in other tests


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_BS():
    check_iterations(num_nodes=[6], ionic_model_name="BS", expected_avg_niters=3.175)


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_BS():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="BS", expected_avg_niters=2.03125)


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_HH():
    check_iterations(num_nodes=[6], ionic_model_name="HH", expected_avg_niters=3.90625)


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_HH():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="HH", expected_avg_niters=2.80625)


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_CRN():
    check_iterations(num_nodes=[6], ionic_model_name="CRN", expected_avg_niters=3.18125)


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_CRN():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="CRN", expected_avg_niters=2.3625)


# @pytest.mark.monodomain
# def test_monodomain_iterations_ESDC_TTP():
#     check_iterations(num_nodes=[6], ionic_model_name="TTP", expected_avg_niters=3.60625)


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_TTP():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="TTP", expected_avg_niters=2.90625)
