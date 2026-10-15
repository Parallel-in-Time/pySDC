import pytest


@pytest.mark.monodomain
@pytest.mark.parametrize('method, order', [('RL1', 1), ('RL2', 2), ('RL2_RDR', 2)])
def test_rush_larsen_order(method, order):
    import numpy as np
    from pySDC.projects.Monodomain.run_scripts.work_precision import run

    # smoothed ionic model, and short enough that the stimulus stays on: the solution is smooth in time
    kw = dict(domain='cuboid_1D_small', refinements=0, model='TTP_SMOOTH', Tend=1.0)
    ref = run('ESDC', 0.01, M=6, sweeps=12, **kw)['uend']
    err = [np.linalg.norm(run(method, dt, **kw)['uend'] - ref) for dt in [0.025, 0.0125]]
    assert np.log2(err[0] / err[1]) > order - 0.15
