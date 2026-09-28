import runpy

import pytest


@pytest.mark.benchmark
def test_B(benchmark):
    import matplotlib.pyplot as plt

    # tutorial step 5 B is a notebook-style script: running it is the benchmark, including its one figure
    benchmark(runpy.run_module, 'pySDC.tutorial.step_5.B_my_first_PFASST_run', run_name='__main__')
    plt.close('all')
