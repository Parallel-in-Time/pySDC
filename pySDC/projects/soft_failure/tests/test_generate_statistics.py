import pytest


@pytest.mark.base
def test_generate_statistics():
    import numpy as np
    from pySDC.projects.soft_failure.generate_statistics import main

    np.random.seed(0)  # the faults are drawn from numpy's global generator
    # the injected bitflips make values overflow or nan, which is what the statistics are about
    with pytest.warns(RuntimeWarning):
        main()
