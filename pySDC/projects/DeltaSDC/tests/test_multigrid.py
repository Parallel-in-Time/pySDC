import numpy as np
import pytest


@pytest.mark.base
@pytest.mark.parametrize('fmt, rtol', [('fp64', 1e-10), ('fp32', 1e-5), ('fp16', 1e-2)])
def test_heat3d_mg(fmt, rtol):
    """The right-hand side is nu times the Laplacian, and a node solve returns its result in the format, to rtol."""
    from pySDC.projects.DeltaSDC.multigrid import DTYPES, heat3d_mg

    nu, factor = 0.5, 1e-3
    prob = heat3d_mg(N=16, nu=nu, solve_format=fmt, mg_rtol=rtol)
    u = prob.u_exact(0.0)
    lap = prob.multigrid.laplacian(np.asarray(u))
    assert np.allclose(prob.eval_f(u, 0.0), nu * lap)
    assert np.allclose(prob.eval_f_increment(u, u, 0.0), nu * lap)  # linear: f(u + d) - f(u) = f(d)

    x = prob.solve_system(u, factor, u, 0.0)
    assert np.asarray(x).dtype == DTYPES[fmt]
    x = np.asarray(x, dtype=np.float64)
    residual = np.asarray(u) - (x - factor * nu * prob.multigrid.laplacian(x))
    assert np.linalg.norm(residual) < 2 * rtol * np.linalg.norm(u)
    assert prob.mg_iterations[-1] > 0
