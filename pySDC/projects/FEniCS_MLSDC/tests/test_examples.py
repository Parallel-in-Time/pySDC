import pytest

from pySDC.projects.FEniCS_MLSDC.setups import EXAMPLES

#: keep the runs small enough for CI; the physics is unchanged, only the number of steps
SHORT = {'nsteps': 4}


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
def test_mlsdc_beats_sdc(example):
    """MLSDC must cost less than SDC, with 'work' charging each coarse level for its dofs."""
    from pySDC.projects.FEniCS_MLSDC.run_examples import compare_mlsdc
    from pySDC.projects.FEniCS_MLSDC.setups import get_tolerance

    results = compare_mlsdc(example, out=lambda *a: None, **SHORT)
    sdc = results[1]

    for nlevels in (2, 3):
        res = results[nlevels]
        assert res['niter'] < sdc['niter'], (
            f'{example}: MLSDC with {nlevels} levels needed {res["niter"]} iterations, '
            f'SDC only {sdc["niter"]}'
        )
        assert res['work'] < sdc['work'], f'{example}: {nlevels} levels cost more than SDC'
        assert abs(res['uend'] - sdc['uend']) < get_tolerance(example), (
            f'{example}: {nlevels}-level MLSDC disagrees with SDC'
        )

    assert len(results[3]['dofs']) == 3 and results[3]['dofs'][-1] < results[3]['dofs'][0]


@pytest.mark.fenics
@pytest.mark.parametrize('example', EXAMPLES)
def test_pfasst_iterations_stay_bounded(example):
    """PFASST must converge and agree with serial over the step counts each example supports.

    The bound is loose on purpose: the examples grow by 1.3-1.7x out to 8 parallel steps.
    """
    from pySDC.projects.FEniCS_MLSDC.run_examples import check_pfasst
    from pySDC.projects.FEniCS_MLSDC.setups import get_pfasst_procs, get_tolerance

    procs = tuple(p for p in get_pfasst_procs(example) if p <= 4)
    results = check_pfasst(example, procs=procs, out=lambda *a: None, **SHORT)
    serial = results[procs[0]]

    for p in procs:
        res = results[p]
        assert res['niter'] <= 3.0 * serial['niter'], (
            f'{example}: PFASST on {p} steps needed {res["niter"]} iterations against '
            f'{serial["niter"]} in serial'
        )
        assert abs(res['uend'] - serial['uend']) < get_tolerance(example), (
            f'{example}: PFASST on {p} steps disagrees with the serial run'
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
    monkeypatch.setattr(
        run_examples, 'compare_mlsdc', lambda example, out=print: compare_mlsdc(example, out=out, **SHORT)
    )
    monkeypatch.setattr(
        run_examples,
        'check_pfasst',
        lambda example, out=print: check_pfasst(example, procs=(1, 2), out=out, **SHORT),
    )
    monkeypatch.chdir(tmp_path)

    run_examples.main()

    report = tmp_path / 'data' / 'fenics_mlsdc_out.txt'
    assert report.exists()
    text = report.read_text()
    assert 'heat' in text and 'PFASST' in text and 'mass-matrix' in text
