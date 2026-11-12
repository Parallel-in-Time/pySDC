import pytest


@pytest.mark.monodomain
def test_parallel_sweep_converges_to_the_serial_collocation_solution():
    import numpy as np
    from pySDC.projects.Monodomain.run_scripts.work_precision import run

    kw = dict(domain='cuboid_1D_small', refinements=0, model='TTP_SMOOTH', Tend=1.0, M=4, restol=1e-12)
    serial = run('ESDC', 0.1, **kw)
    parallel = run('ESDC_par', 0.1, **kw)
    assert np.max(np.abs(parallel['uend'] - serial['uend'])) < 1e-8
