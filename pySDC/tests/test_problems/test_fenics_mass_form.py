import pytest


def _grayscott_pair(**kwargs):
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import (
        fenics_grayscott,
        fenics_grayscott_mass,
    )

    params = dict(c_nvars=32, family='CG', order=4, refinements=1, newton_tol=1e-12, newton_rtol=1e-11)
    params.update(kwargs)
    return fenics_grayscott(**params), fenics_grayscott_mass(**params)


@pytest.mark.fenics
def test_apply_mass_matrix_has_no_silent_default():
    """A problem used in a mass formulation without implementing it must fail, not return u."""
    from pySDC.core.problem import Problem

    class NoMass(Problem):
        pass

    with pytest.raises(NotImplementedError, match='apply_mass_matrix'):
        NoMass.apply_mass_matrix(NoMass.__new__(NoMass), None)


@pytest.mark.fenics
def test_mass_form_matches_mass_inverse():
    """The two formulations must agree method by method: f_mass == M f_minv, and the solves agree.

    This is the check that catches a missing or wrong apply_mass_matrix; the failure is invisible
    at the integrator level, where it only shows up as a stalled iteration many sweeps later.
    """
    import numpy as np

    mi, ma = _grayscott_pair()
    u = mi.u_exact(0.0)

    f_mi, f_ma = mi.eval_f(u, 0.0), ma.eval_f(u, 0.0)
    Mf = mi.apply_mass_matrix(f_mi)
    scale = np.abs(f_ma.values.vector()[:]).max()
    err = np.abs(f_ma.values.vector()[:] - Mf.values.vector()[:]).max()
    assert err < 1e-10 * scale, f'eval_f mismatch between formulations: {err:.3e} (scale {scale:.3e})'

    factor = 1e-2 / 3
    r = mi.u_exact(0.0)
    s_mi = mi.solve_system(r, factor, u, 0.0)
    s_ma = ma.solve_system(mi.apply_mass_matrix(r), factor, u, 0.0)
    diff = np.abs(s_ma.values.vector()[:] - s_mi.values.vector()[:]).max()
    assert diff < 1e-10, f'solve_system mismatch between formulations: {diff:.3e}'


@pytest.mark.fenics
def test_transfer_operators():
    """P must reproduce df.interpolate, and P^T must give the Galerkin coarse operators exactly."""
    import numpy as np
    import dolfin as df
    from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass

    fine = fenics_heat_mass(c_nvars=32, family='CG', order=1, refinements=1, t0=0.0)
    coarse = fenics_heat_mass(c_nvars=32, family='CG', order=1, refinements=0, t0=0.0)
    st = mesh_to_mesh_fenics(fine_prob=fine, coarse_prob=coarse, params={})

    P = st.Pmat.toarray()
    u_c = df.interpolate(df.Expression('sin(pi*x[0])', degree=4, pi=np.pi), coarse.V)
    assert np.abs(P @ u_c.vector()[:] - df.interpolate(u_c, fine.V).vector()[:]).max() < 1e-12

    Mf, Mc = fine.M.array(), coarse.M.array()
    assert np.abs(P.T @ Mf @ P - Mc).max() < 1e-12, 'P^T M_f P != M_c'

    w = df.interpolate(df.Expression('sin(3*pi*x[0])', degree=6, pi=np.pi), fine.V)
    dual = st.restrict_dual(fine.dtype_u(w))
    assert np.abs(dual.values.vector()[:] - P.T @ w.vector()[:]).max() < 1e-12, 'restrict_dual != P^T'


def _run_mass(nlevels, dt=1.0, maxiter=30):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

    description = {
        'problem_class': fenics_grayscott_mass,
        'problem_params': {
            'c_nvars': 32,
            'family': 'CG',
            'order': 4,
            'refinements': [2, 1, 0][:nlevels],
            'newton_tol': 1e-12,
            'newton_rtol': 1e-11,
        },
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3] * nlevels, 'QI': 'LU'},
        'level_params': {'restol': 1e-9, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    if nlevels > 1:
        description['space_transfer_class'] = mesh_to_mesh_fenics
        description['space_transfer_params'] = {}
        description['base_transfer_class'] = base_transfer_mass
        description['base_transfer_params'] = {'finter': False}

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2 * dt)
    niter = np.mean([v for (_, v) in get_sorted(stats, type='niter', sortby='time')])
    return niter, uend


@pytest.mark.fenics
def test_mlsdc_mass_form_converges():
    """SDC, 2-level and 3-level MLSDC in mass form all converge to the same solution.

    Three levels is here on purpose: it used to hit maxiter, because u0 was only put into the dual
    space on the finest level, so the 1 -> 2 transfer L2-projected an already-dual vector.
    """
    maxiter = 30
    results = {n: _run_mass(n, maxiter=maxiter) for n in (1, 2, 3)}

    for n, (niter, _) in results.items():
        assert niter < maxiter, f'{n}-level run did not converge, hit maxiter ({niter})'

    sdc_niter, sdc_u = results[1]
    for n in (2, 3):
        niter, u = results[n]
        assert niter <= sdc_niter, f'{n}-level MLSDC needed more iterations than SDC: {niter} vs {sdc_niter}'
        assert abs(u - sdc_u) < 1e-7, f'{n}-level MLSDC disagrees with SDC on the solution'
