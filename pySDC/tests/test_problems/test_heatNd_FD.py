import pytest


@pytest.mark.base
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_u_exact_solves_semi_discrete_problem(ndim):
    """
    With the second-order stencil, u_exact must be an exact solution of the semi-discrete problem du/dt = A u.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    prob = heatNd_unforced(nvars=(16,) * ndim, freq=(2,) * ndim, nu=0.1, order=2, bc='periodic')

    t = 0.3
    u = prob.u_exact(t)
    # u_exact(t) = exp(-nu rho t) u_exact(0), so du/dt = log(u(t)/u(0)) / t * u(t)
    dudt = np.log(np.max(u) / np.max(prob.u_exact(0))) / t * u
    assert np.allclose(prob.eval_f(u, t), dudt, atol=1e-12)


def spatial_residual(prob, t, h=1e-5):
    """Max norm of the right-hand side at the exact solution minus its centred time derivative."""
    import numpy as np

    f = prob.eval_f(prob.u_exact(t), t)
    rhs = np.asarray(f.impl + f.expl) if hasattr(f, 'impl') else np.asarray(f)
    dudt = (np.asarray(prob.u_exact(t + h)) - np.asarray(prob.u_exact(t - h))) / (2 * h)
    return np.max(np.abs(rhs - dudt))


@pytest.mark.base
@pytest.mark.parametrize(
    'name, ndim, freq, nvars',
    [
        ('unforced', 1, -1, (64, 128, 256)),
        ('forced', 1, 2, (32, 64, 128)),
        ('forced', 2, 2, (32, 64, 128)),
        ('forced', 3, 2, (16, 32, 64)),
    ],
)
@pytest.mark.parametrize('order', [2, 4])
def test_u_exact_converges_in_space(name, ndim, freq, nvars, order):
    """
    u_exact must solve the PDE, so plugging it into the semi-discrete right-hand side leaves only the
    spatial discretization error, which has to fall at the stencil order.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced, heatNd_unforced

    problem_class = heatNd_forced if name == 'forced' else heatNd_unforced
    errors = [
        spatial_residual(problem_class(nvars=(n,) * ndim, freq=(freq,) * ndim, nu=0.1, order=order, bc='periodic'), 0.3)
        for n in nvars
    ]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > order - 0.3), f'expected order {order}, got {rates} from errors {errors}'
