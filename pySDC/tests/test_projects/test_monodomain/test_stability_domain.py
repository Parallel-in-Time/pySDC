import pytest


@pytest.mark.monodomain
def test_stability_ESDC():
    from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

    main(dl=100, openmp=False, n_time_ranks=1, end_time=1.0, num_nodes=[5])


@pytest.mark.monodomain
def test_stability_MLESDC():
    from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

    main(dl=100, openmp=False, n_time_ranks=1, end_time=1.0, num_nodes=[5, 3])


@pytest.mark.monodomain
def test_stability_PFASST():
    from pySDC.projects.Monodomain.run_scripts.run_TestODE import main

    main(dl=100, openmp=False, n_time_ranks=4, end_time=4.0, num_nodes=[5, 3])


# if __name__ == "__main__":
#     test_stability_ESDC()
#     test_stability_MLESDC()
#     test_stability_PFASST()
