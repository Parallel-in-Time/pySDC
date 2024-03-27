import pytest


@pytest.mark.monodomain
def test_monodomain_stability_ESDC():
    from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

    main(
        integrator="IMEXEXP_EXPRK",
        dl=2,
        l_min=-100,
        openmp=True,
        n_time_ranks=1,
        end_time=1.0,
        num_nodes=[5],
        check_stability=True,
    )

    # This is to generate the image only, we do not check for stabiltiy since we already know that
    # SDC is unstable for this problem
    main(
        integrator="IMEXEXP",
        dl=2,
        l_min=-100,
        openmp=True,
        n_time_ranks=1,
        end_time=1.0,
        num_nodes=[5],
        check_stability=False,
    )


# @pytest.mark.monodomain
# def test_monodomain_stability_MLESDC():
#     from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

#     main(
#         integrator="IMEXEXP_EXPRK",
#         dl=2,
#         l_min=-100,
#         openmp=True,
#         n_time_ranks=1,
#         end_time=1.0,
#         num_nodes=[5, 3],
#         check_stability=True,
#     )


# @pytest.mark.monodomain
# def test_monodomain_stability_PFASST():
#     from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

#     main(
#         integrator="IMEXEXP_EXPRK",
#         dl=2,
#         l_min=-100,
#         openmp=True,
#         n_time_ranks=4,
#         end_time=1.0,
#         num_nodes=[5, 3],
#         check_stability=True,
#     )


# if __name__ == "__main__":
#     test_monodomain_stability_ESDC()
#     test_monodomain_stability_MLESDC()
#     test_monodomain_stability_PFASST()
