import pytest


@pytest.mark.benchmark
def test_B(benchmark):
    from pySDC.tutorial.step_5.B_my_first_PFASST_run import main as main_B

    benchmark(main_B)
