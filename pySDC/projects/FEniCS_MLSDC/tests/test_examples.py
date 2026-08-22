import pytest

from pySDC.projects.FEniCS_MLSDC.setups import EXAMPLES, FAMILIES

#: keep the runs small enough for CI; the physics is unchanged, only the number of steps
SHORT = {'nsteps': 4}


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.parametrize('family', FAMILIES)
def test_mlsdc_beats_sdc(example, family):
    """MLSDC on a coarser mesh must cost less than SDC, with 'work' charging each level for its dofs.

    This holds for DG only because the hierarchy is a proper Galerkin one: an exact prolongation,
    and an interior penalty pinned across levels. Get either wrong and DG still converges to the
    right answer, just with a coarse level that corrects almost nothing.
    """
    from pySDC.projects.FEniCS_MLSDC.run_examples import compare_mlsdc
    from pySDC.projects.FEniCS_MLSDC.setups import get_tolerance

    results = compare_mlsdc(example, family=family, coarsening='h', out=lambda *a: None, **SHORT)
    sdc = results[1]

    for nlevels in (2, 3):
        res = results[nlevels]
        assert res['niter'] < sdc['niter'], (
            f'{example} [{family}]: MLSDC with {nlevels} levels needed {res["niter"]} iterations, '
            f'SDC only {sdc["niter"]}'
        )
        assert abs(res['uend'] - sdc['uend']) < get_tolerance(example), (
            f'{example} [{family}]: {nlevels}-level MLSDC disagrees with SDC'
        )
        assert res['work'] < sdc['work'], f'{example} [{family}]: {nlevels} levels cost more than SDC'

    assert len(results[3]['dofs']) == 3 and results[3]['dofs'][-1] < results[3]['dofs'][0]


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.parametrize('family', FAMILIES)
def test_h_coarsening_beats_p_coarsening(example, family):
    """Coarsen the mesh, not the element order.

    Both ladders are nested and converge to the same answer, but dropping the order leaves a coarse
    space that approximates the smooth part of the error far worse, so it buys fewer iterations.
    This is 'use high-order elements' again, now pointed at the coarsening direction.
    """
    from pySDC.projects.FEniCS_MLSDC.run_examples import compare_mlsdc
    from pySDC.projects.FEniCS_MLSDC.setups import get_tolerance

    h = compare_mlsdc(example, family=family, coarsening='h', out=lambda *a: None, **SHORT)
    p = compare_mlsdc(example, family=family, coarsening='p', out=lambda *a: None, **SHORT)

    assert h[1]['work'] == pytest.approx(p[1]['work']), 'SDC must not depend on the coarsening'

    for nlevels in (2, 3):
        assert abs(p[nlevels]['uend'] - h[1]['uend']) < get_tolerance(example), (
            f'{example} [{family}]: {nlevels}-level p-coarsening disagrees with SDC'
        )
        assert h[nlevels]['work'] <= p[nlevels]['work'], (
            f'{example} [{family}]: p-coarsening cost {p[nlevels]["work"]} against {h[nlevels]["work"]} '
            f'for h-coarsening on {nlevels} levels'
        )


@pytest.mark.fenics
@pytest.mark.parametrize('coarsening', ['h', 'p'])
def test_prolongation_preserves_jumps(coarsening):
    """The DG prolongation must be the inclusion, jumps and all.

    dolfin's cross-mesh ``interpolate`` hands a fine dof sitting on a coarse facet whichever of its
    two coarse values the bounding-box tree finds first, quietly continuising the coarse function.
    The error is O(1) in the jump and invisible for smooth data -- so test it with data that is not
    smooth.
    """
    import numpy as np

    from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
    from pySDC.projects.FEniCS_MLSDC.setups import get_description

    problem_class = get_description('heat', family='DG')[0]['problem_class']
    params = dict(c_nvars=16, family='DG', order=4, t0=0.0)
    fine = problem_class(refinements=1 if coarsening == 'h' else 0, **params)
    coarse = problem_class(refinements=0, **dict(params, order=4 if coarsening == 'h' else 2))

    u = coarse.dtype_u(coarse.V)
    u.values.vector()[:] = np.random.default_rng(0).normal(size=coarse.V.dim())

    prolonged = mesh_to_mesh_fenics(fine_prob=fine, coarse_prob=coarse, params={}).prolong(u)

    xs = np.linspace(1e-7, 1 - 1e-7, 301)
    err = max(abs(prolonged.values(x) - u.values(x)) for x in xs)
    assert err < 1e-12, f'{coarsening}-prolongation is not the inclusion: off by {err:.3e}'


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
def test_dg_and_cg_solve_the_same_problem(example):
    """The DG forms are a discretisation, not a different model: at order 4 the two agree closely.

    This is the check that keeps the interior penalty and the Nitsche and Lax-Friedrichs terms
    honest -- a wrong sign or a missing facet term shows up here and nowhere else.
    """
    import dolfin as df

    from pySDC.projects.FEniCS_MLSDC.run_examples import run

    cg, dg = (run(example, family=f, **SHORT)['uend'] for f in ('CG', 'DG'))

    V = cg.values.function_space()
    diff = df.Function(V)
    diff.vector()[:] = cg.values.vector()[:] - df.interpolate(dg.values, V).vector()[:]
    rel = abs(df.assemble(df.inner(diff, diff) * df.dx) / df.assemble(df.inner(cg.values, cg.values) * df.dx)) ** 0.5

    assert rel < 1e-6, f'{example}: CG and DG differ by {rel:.2e} relative'


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.parametrize('family', FAMILIES)
def test_pfasst_iterations_stay_bounded(example, family):
    """PFASST must converge and agree with serial over the step counts each example supports.

    The bound is loose on purpose: the examples grow by 1.3-1.9x out to 8 parallel steps, with DG
    within a whisker of CG.
    """
    from pySDC.projects.FEniCS_MLSDC.run_examples import check_pfasst
    from pySDC.projects.FEniCS_MLSDC.setups import get_pfasst_procs, get_tolerance

    procs = tuple(p for p in get_pfasst_procs(example) if p <= 4)
    results = check_pfasst(example, family=family, procs=procs, out=lambda *a: None, **SHORT)
    serial = results[procs[0]]

    for p in procs:
        res = results[p]
        assert res['niter'] <= 3.0 * serial['niter'], (
            f'{example} [{family}]: PFASST on {p} steps needed {res["niter"]} iterations against '
            f'{serial["niter"]} in serial'
        )
        assert abs(res['uend'] - serial['uend']) < get_tolerance(example), (
            f'{example} [{family}]: PFASST on {p} steps disagrees with the serial run'
        )


@pytest.mark.fenics
def test_run_reports_dofs_and_work():
    from pySDC.projects.FEniCS_MLSDC.run_examples import run

    res = run('heat', nlevels=2, **SHORT)
    assert res['dofs'][1] < res['dofs'][0]
    assert res['work'] == pytest.approx(res['niter'] * sum(n / res['dofs'][0] for n in res['dofs']))


@pytest.mark.fenics
def test_main_writes_a_report(tmp_path, monkeypatch):
    """Cover main() end to end, shrunk to the cheapest example."""
    import pySDC.projects.FEniCS_MLSDC.run_examples as run_examples

    compare_mlsdc, check_pfasst = run_examples.compare_mlsdc, run_examples.check_pfasst

    monkeypatch.setattr(run_examples, 'EXAMPLES', ('heat',))
    monkeypatch.setattr(run_examples, 'FAMILIES', ('DG',))
    monkeypatch.setattr(run_examples, 'COARSENINGS', ('h',))
    monkeypatch.setattr(
        run_examples,
        'compare_mlsdc',
        lambda example, out=print, **kw: compare_mlsdc(example, out=out, **kw, **SHORT),
    )
    monkeypatch.setattr(
        run_examples,
        'check_pfasst',
        lambda example, out=print, **kw: check_pfasst(example, procs=(1, 2), out=out, **kw, **SHORT),
    )
    monkeypatch.chdir(tmp_path)

    run_examples.main()

    report = tmp_path / 'data' / 'fenics_mlsdc_out.txt'
    assert report.exists()
    text = report.read_text()
    assert 'heat' in text and 'PFASST' in text and 'mass-matrix' in text and '[DG, h]' in text
