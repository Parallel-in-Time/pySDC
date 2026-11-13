import pytest

SMALL = dict(N=256, L=51.2, x_cap=18.0, eta_cap=1e-2, E0=0.02, ncycles=2)


@pytest.mark.base
def test_ground_state():
    import numpy as np
    from pySDC.projects.TDDFT.problem import KohnShamMolecule1D

    P = KohnShamMolecule1D(**SMALL)
    psi = np.asarray(P.ground_state())
    assert abs(P.norm(psi) - P.norb) < 1e-12
    assert np.allclose(psi.conj() @ psi.T * P.dx, np.eye(P.norb), atol=1e-12)
    assert abs(P.dipole(psi)) < 1e-8, 'the molecule is symmetric'
    assert P.eps[P.norb] - P.eps[P.norb - 1] > 0.2, 'occupied and empty orbitals are separated by a gap'
    assert all(c.niter == 0 for c in P.work_counters.values()), 'the setup is not counted'


@pytest.mark.base
def test_projector_exponential():
    import numpy as np
    import scipy.linalg as sl
    from pySDC.projects.TDDFT.problem import KohnShamMolecule1D
    from pySDC.projects.TDDFT.propagators import projector_exp

    P = KohnShamMolecule1D(**SMALL)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((P.norb, P.N)) + 1j * rng.standard_normal((P.norb, P.N))
    Vnl = P.E_kb * P.beta.T @ P.beta * P.dx
    assert np.allclose(P.apply_nonlocal(psi), psi @ Vnl.T, atol=1e-13)
    assert np.allclose(projector_exp(P, psi, 0.3), psi @ sl.expm(-0.3j * Vnl).T, atol=1e-13)


@pytest.mark.base
def test_implicit_part_reused_after_solve():
    import numpy as np
    from pySDC.projects.TDDFT.problem import KohnShamMolecule1D

    P = KohnShamMolecule1D(**SMALL)
    u0 = P.ground_state()
    u = P.solve_system(u0, 0.1, u0, 0.0)
    n = P.work_counters['fft'].niter
    reused = P.eval_f(u, 0.0).impl
    assert P.work_counters['fft'].niter == n, 'no FFT for the implicit part right after the solve'
    assert np.allclose(reused, P.eval_f(P.dtype_u(u), 0.0).impl, atol=1e-12)


def _dipole_errors(method, dts, **kw):
    import numpy as np
    from pySDC.projects.TDDFT.propagators import run_propagator
    from pySDC.projects.TDDFT.sdc import run_sdc

    T = 8.0
    ref = run_sdc(0.125, T, 1.0, M=5, restol=1e-12, problem_params=SMALL)['d']
    return [np.max(np.abs(run_propagator(method, dt, T, 1.0, SMALL, **kw)['d'] - ref)) for dt in dts]


@pytest.mark.base
@pytest.mark.parametrize('method, order', [('strang', 2), ('aetrs', 2), ('crank_nicolson', 2), ('cfm4', 4)])
def test_propagator_order(method, order):
    import numpy as np

    dts = [0.1, 0.05] if method != 'aetrs' else [0.02, 0.01]  # Taylor-4 exponentials need small steps
    err = _dipole_errors(method, dts)
    assert np.log2(err[0] / err[1]) > order - 0.3, err


@pytest.mark.base
def test_parallel_sweep_converges_to_collocation():
    import numpy as np
    from pySDC.projects.TDDFT.sdc import run_sdc

    kw = dict(dt=0.25, Tend=8.0, dt_out=1.0, M=4, restol=1e-12, problem_params=SMALL)
    serial = run_sdc(**kw)
    parallel = run_sdc(**kw, QI='MIN-SR-S', QE='PIC')
    assert np.max(np.abs(serial['d'] - parallel['d'])) < 1e-10, 'both sweeps converge to the same collocation solution'
    assert parallel['cost'] < parallel['cost_total'] / 3, 'the critical path counts one node per sweep'


@pytest.mark.base
def test_work_precision_demo():
    """The demo behind the gallery figure: parallel SDC is cheaper than Magnus 4 at the same dipole error"""
    import os
    import numpy as np
    from pySDC.projects.TDDFT.work_precision import main

    rows = main()
    assert os.path.isfile('data/tddft_work_precision.png')

    def cheapest(label, tol):
        return min([r['cost'] for r in rows if r['label'] == label and r['err'] <= tol], default=np.inf)

    assert cheapest('sdc_parallel', 1e-5) * 2 < cheapest('cfm4', 1e-5)
