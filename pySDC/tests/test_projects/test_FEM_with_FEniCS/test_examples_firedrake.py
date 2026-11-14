"""
The claims of ``pySDC/projects/FEM_with_FEniCS/tests/test_examples.py``, for the Firedrake study
(:mod:`pySDC.projects.FEM_with_FEniCS.run_examples_firedrake`). Separate runs build separate (identical) meshes, so
solutions are compared through their dof vectors, which identical constructions number alike.
"""

from functools import cache

import pytest

#: keep the runs small enough for CI; the physics is unchanged, only the number of steps (the FEniCS tests run 4)
SHORT = {'nsteps': 2}

SAVINGS = [(e, f) for e in ('heat', 'burgers', 'grayscott') for f in ('CG', 'DG')]
CASES = SAVINGS + [('vortex', 'CG')]


def l2_diff(u, v):
    """L2 norm of the difference of two solutions of identically built problems."""
    import firedrake as fd

    w = fd.Function(u.functionspace.function_space())
    with v.dat.vec_ro as source, w.dat.vec_wo as target:
        source.copy(target)
    return fd.norm(u.functionspace - w)


@cache
def mlsdc(example, family, coarsening):
    """SDC and MLSDC on 2 and 3 levels, shared by the tests that compare them."""
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import compare_mlsdc

    return compare_mlsdc(example, family=family, coarsening=coarsening, out=lambda *a: None, **SHORT)


@pytest.mark.firedrake
@pytest.mark.parametrize('example, family', SAVINGS)
def test_mlsdc_beats_sdc(example, family):
    """MLSDC on a coarser mesh must cost less than SDC, with 'work' charging each level for its dofs."""
    from pySDC.projects.FEM_with_FEniCS.setups_firedrake import get_tolerance

    results = mlsdc(example, family, 'h')
    sdc = results[1]

    for nlevels in (2, 3):
        res = results[nlevels]
        assert res['niter'] < sdc['niter'], (example, family, nlevels, res['niter'], sdc['niter'])
        assert l2_diff(res['uend'], sdc['uend']) < get_tolerance(example), (example, family, nlevels)
        assert res['work'] < sdc['work'], (example, family, nlevels, res['work'], sdc['work'])

    assert len(results[3]['dofs']) == 3 and results[3]['dofs'][-1] < results[3]['dofs'][0]


@pytest.mark.firedrake
@pytest.mark.parametrize('example, family', SAVINGS)
def test_h_coarsening_beats_p_coarsening(example, family):
    """Coarsen the mesh, not the element order: both converge, but the lower order buys fewer iterations."""
    from pySDC.projects.FEM_with_FEniCS.setups_firedrake import get_tolerance

    h, p = mlsdc(example, family, 'h'), mlsdc(example, family, 'p')

    assert h[1]['work'] == pytest.approx(p[1]['work']), 'SDC must not depend on the coarsening'

    for nlevels in (2, 3):
        assert l2_diff(p[nlevels]['uend'], h[1]['uend']) < get_tolerance(example), (example, family, nlevels)
        assert h[nlevels]['work'] <= p[nlevels]['work'], (example, family, nlevels, h[nlevels]['work'])


@pytest.mark.firedrake
@pytest.mark.parametrize('coarsening', ['h', 'p'])
def test_prolongation_preserves_jumps(coarsening):
    """The DG prolongation must be the inclusion, jumps and all; tested with data that is not smooth."""
    from types import SimpleNamespace

    import firedrake as fd
    import numpy as np
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import (
        MeshToMeshFiredrake,
        MeshToMeshFiredrakeHierarchy,
    )

    coarse_mesh, fine_mesh = fd.MeshHierarchy(fd.UnitIntervalMesh(16), 1)
    if coarsening == 'h':
        spaces = [fd.FunctionSpace(m, 'DG', 4) for m in (fine_mesh, coarse_mesh)]
        transfer = MeshToMeshFiredrakeHierarchy
    else:
        spaces = [fd.FunctionSpace(coarse_mesh, 'DG', p) for p in (4, 2)]
        transfer = MeshToMeshFiredrake
    fine, coarse = [SimpleNamespace(init=V, dtype_u=firedrake_mesh, u_init=firedrake_mesh(V)) for V in spaces]

    u = firedrake_mesh(fd.Function(coarse.init))
    u.dat.data[:] = np.random.default_rng(0).normal(size=u.dat.data.shape)
    prolonged = transfer(fine, coarse, {}).prolong(u)

    points = [[x] for x in np.linspace(1e-7, 1 - 1e-7, 301)]
    values = [
        fd.PointEvaluator(V.mesh(), points).evaluate(w.functionspace)
        for V, w in zip(spaces, (prolonged, u), strict=True)
    ]
    err = np.abs(values[0] - values[1]).max()
    assert err < 1e-12, f'{coarsening}-prolongation is not the inclusion: off by {err:.3e}'


@pytest.mark.firedrake
@pytest.mark.parametrize('example', ['heat', 'burgers', 'grayscott'])
def test_dg_and_cg_solve_the_same_problem(example):
    """
    The DG forms are a discretisation, not a different model: at order 4 the two agree closely. This keeps the interior
    penalty and the Nitsche and Lax-Friedrichs terms honest.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import run

    cg, dg = (run(example, family=f, **SHORT)['uend'] for f in ('CG', 'DG'))

    def samples(u):
        parts = u.functionspace.subfunctions
        mesh = parts[0].function_space().mesh()
        lo, hi = mesh.coordinates.dat.data_ro.min(), mesh.coordinates.dat.data_ro.max()
        points = [[x] for x in np.linspace(lo, hi, 1001)[1:-1]]
        return np.concatenate([fd.PointEvaluator(mesh, points).evaluate(p) for p in parts])

    # relative RMS over samples in place of the FEniCS version's L2 norm, which needs both on one mesh
    a, b = samples(cg), samples(dg)
    rel = np.sqrt(np.mean((a - b) ** 2) / np.mean(a**2))
    assert rel < 1e-6, f'{example}: CG and DG differ by {rel:.2e} relative'


@pytest.mark.firedrake
@pytest.mark.parametrize('example, family', CASES)
def test_pfasst_iterations_stay_bounded(example, family):
    """PFASST must converge and agree with serial over the step counts each example supports."""
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import check_pfasst
    from pySDC.projects.FEM_with_FEniCS.setups_firedrake import get_pfasst_procs, get_tolerance

    # the vortex is 2D and solves a Poisson problem per evaluation: 2 parallel steps keep it under 20 s
    nmax = 2 if example == 'vortex' else 4
    procs = tuple(p for p in get_pfasst_procs(example) if p <= nmax)
    results = check_pfasst(example, family=family, procs=procs, out=lambda *a: None, nsteps=nmax)
    serial = results[procs[0]]

    for p in procs:
        res = results[p]
        assert res['niter'] <= 3.0 * serial['niter'], (example, family, p, res['niter'], serial['niter'])
        assert l2_diff(res['uend'], serial['uend']) < get_tolerance(example), (example, family, p)


@pytest.mark.firedrake
def test_run_reports_dofs_and_work():
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import RESTRICTION_COST, run

    res = run('heat', nlevels=2, **SHORT)
    assert res['dofs'][1] < res['dofs'][0]

    # work charges the sweeps on every level plus the solution restrictions, one per coarse level per iteration
    num_nodes = 3
    restrictions = (num_nodes + 1) * (len(res['dofs']) - 1) * RESTRICTION_COST
    expected = res['niter'] * (sum(n / res['dofs'][0] for n in res['dofs']) + restrictions)
    assert res['work'] == pytest.approx(expected)
    assert res['work'] > res['niter'] * sum(n / res['dofs'][0] for n in res['dofs']), 'transfer charged nothing'


@pytest.mark.firedrake
def test_high_order_elements_pay(tmp_path):
    """
    The headline: at equal dofs per level, the speed-up of MLSDC grows with the element order. At CG4 every savings
    example pays on two and on three levels, and more than at CG1. Plots it as the FEniCS test does for the gallery.
    """
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import compare_orders, plot_orders
    from pySDC.projects.FEM_with_FEniCS.setups_firedrake import get_order_study

    examples = ('heat', 'burgers', 'grayscott')
    # the claim compares the lowest and the highest order; the study's middle one (CG2) only adds runtime
    orders = {e: (min(get_order_study(e)), max(get_order_study(e))) for e in examples}
    speedups = {e: compare_orders(e, orders=orders[e], out=lambda *a: None, **SHORT) for e in examples}

    for example, study in speedups.items():
        low, high = orders[example]
        for nlevels in (2, 3):
            assert study[(high, nlevels)] > 1, f'{example}: {nlevels} levels at CG{high} do not pay'
            assert study[(high, nlevels)] > study[(low, nlevels)], f'{example}: {nlevels} levels gain less at CG{high}'

    plot_orders(speedups, fname=tmp_path / 'orders.png')
    assert (tmp_path / 'orders.png').stat().st_size > 0


@pytest.mark.firedrake
def test_main_writes_a_report(tmp_path, monkeypatch):
    """Cover main() end to end, shrunk to the cheapest example."""
    import pySDC.projects.FEM_with_FEniCS.run_examples_firedrake as run_examples

    compare_mlsdc, check_pfasst = run_examples.compare_mlsdc, run_examples.check_pfasst

    monkeypatch.setattr(run_examples, 'EXAMPLES', ('heat',))
    monkeypatch.setattr(run_examples, 'get_families', lambda example: ('DG',))
    monkeypatch.setattr(run_examples, 'get_coarsenings', lambda example: ('h',))
    monkeypatch.setattr(run_examples, 'get_order_study', lambda example: ())
    monkeypatch.setattr(
        run_examples, 'compare_mlsdc', lambda example, out=print, **kw: compare_mlsdc(example, out=out, **kw, **SHORT)
    )
    monkeypatch.setattr(
        run_examples,
        'check_pfasst',
        lambda example, out=print, **kw: check_pfasst(example, procs=(1, 2), out=out, **kw, **SHORT),
    )
    monkeypatch.chdir(tmp_path)

    run_examples.main()

    text = (tmp_path / 'data' / 'fem_with_firedrake_out.txt').read_text()
    assert 'heat' in text and 'PFASST' in text and 'mass-matrix' in text and '[DG, h]' in text


@pytest.mark.firedrake
def test_prolongation_on_a_periodic_space():
    """
    The prolongation must be the inclusion on a periodic space too: the FEniCS version extrapolated across the domain
    there. Here the vortex's hierarchy of periodic meshes, with random coarse data.
    """
    from types import SimpleNamespace

    import firedrake as fd
    import numpy as np
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    coarse_mesh, fine_mesh = fd.MeshHierarchy(fd.PeriodicUnitSquareMesh(8, 8, diagonal='right'), 1)
    spaces = [fd.FunctionSpace(m, 'CG', 2) for m in (fine_mesh, coarse_mesh)]
    fine, coarse = [SimpleNamespace(init=V, dtype_u=firedrake_mesh, u_init=firedrake_mesh(V)) for V in spaces]
    transfer = MeshToMeshFiredrakeHierarchy(fine, coarse, {})

    one = transfer.prolong(firedrake_mesh(fd.Function(coarse.init).assign(1.0)))
    assert np.allclose(one.dat.data_ro, 1.0), 'the prolongation does not reproduce a constant'

    u = firedrake_mesh(fd.Function(coarse.init))
    u.dat.data[:] = np.random.default_rng(0).normal(size=u.dat.data.shape)
    prolonged = transfer.prolong(u)
    assert np.abs(prolonged.dat.data_ro).max() <= np.abs(u.dat.data_ro).max() * 1.25 + 1e-12

    points = np.random.default_rng(1).uniform(1e-6, 1 - 1e-6, size=(200, 2)).tolist()
    values = [
        fd.PointEvaluator(V.mesh(), points).evaluate(w.functionspace)
        for V, w in zip(spaces, (prolonged, u), strict=True)
    ]
    err = np.abs(values[0] - values[1]).max()
    assert err < 1e-12, f'periodic prolongation is not the inclusion: off by {err:.3e}'


@pytest.mark.firedrake
def test_a_coarse_level_that_does_not_pay_still_converges():
    """The vortex is not here to save work, but it still has to reach the same answer with a coarse level."""
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import run
    from pySDC.projects.FEM_with_FEniCS.setups_firedrake import get_tolerance, pays_off

    assert not pays_off('vortex')
    sdc = run('vortex', nlevels=1, **SHORT)
    for nlevels in (2, 3):
        res = run('vortex', nlevels=nlevels, **SHORT)
        assert l2_diff(res['uend'], sdc['uend']) < get_tolerance('vortex'), f'{nlevels}-level MLSDC disagrees'
