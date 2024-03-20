import pytest


def check_iterations(expected_avg_niters, **opts):
    from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run

    # define sweeper parameters
    opts["integrator"] = "IMEXEXP_EXPRK"
    opts["num_sweeps"] = [1]

    # set step parameters
    opts["max_iter"] = 100

    # set level parameters
    opts["dt"] = 0.025

    opts["restol"] = 5e-8  # residual tolerance

    # skip residual computation at coarser levels (if any)
    opts["skip_residual_computation"] = True

    # interpolate or recompute rhs on fine level
    opts["finter"] = True

    # set time parallelism to True or emulated (False)
    opts["truly_time_parallel"] = False

    # set monodomain parameters
    opts["order"] = 4  # 2 or 4
    opts["enable_output"] = False
    opts["write_database"] = False

    opts["output_root"] = "results_iterations_pytest"

    opts["read_init_val"] = False
    opts["init_time"] = 0.0
    opts["end_time"] = 2.0
    opts["write_as_reference_solution"] = False
    opts["write_all_variables"] = False
    opts["output_file_name"] = "monodomain"
    opts["ref_sol"] = ""

    err, rel_err, avg_niters, times, niters, residuals = setup_and_run(**opts)

    print(f"Got average number of iterations {avg_niters}, expected was {expected_avg_niters}")

    assert avg_niters == pytest.approx(
        expected_avg_niters, rel=0.1
    ), f"Average number of iterations {avg_niters} too different from the expected {expected_avg_niters}"


# Many of the following are commented since they test features already tested in other tests
# If you reactivate them the number of expected iterations should be updated


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_BS():
    check_iterations(
        domain_name="cuboid_2D_small",
        num_nodes=[6, 3],
        refinements=[0, -1],
        ionic_model_name="BS",
        n_time_ranks=4,
        expected_avg_niters=3.3209876543209877,
    )


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_BS():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="BS", expected_avg_niters=2.03125)


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_HH():
    check_iterations(
        domain_name="cuboid_2D_small",
        num_nodes=[6, 3],
        refinements=[0, -1],
        ionic_model_name="HH",
        n_time_ranks=2,
        expected_avg_niters=3.074074074074074,
    )


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_HH():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="HH", expected_avg_niters=2.80625)


@pytest.mark.monodomain
def test_monodomain_iterations_ESDC_CRN():
    check_iterations(
        domain_name="cube_1D",
        num_nodes=[6],
        refinements=[0],
        ionic_model_name="CRN",
        n_time_ranks=1,
        expected_avg_niters=3.382716,
    )


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_CRN():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="CRN", expected_avg_niters=2.3625)


# @pytest.mark.monodomain
# def test_monodomain_iterations_ESDC_TTP():
#     check_iterations(num_nodes=[6], ionic_model_name="TTP", expected_avg_niters=3.60625)


# @pytest.mark.monodomain
# def test_monodomain_iterations_MLESDC_TTP():
#     check_iterations(num_nodes=[6, 3], ionic_model_name="TTP", expected_avg_niters=2.90625)


# if __name__ == "__main__":
# test_monodomain_iterations_ESDC_BS()
# test_monodomain_iterations_ESDC_HH()
# test_monodomain_iterations_ESDC_CRN()
