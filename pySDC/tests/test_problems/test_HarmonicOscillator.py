import pytest


@pytest.mark.base
@pytest.mark.parametrize('mu', [0.0, 0.5, 2.0, 3.0])  # undamped, underdamped, critically damped, overdamped
def test_u_exact_solves_ode(mu):
    """u_exact must start at u0 and satisfy x' = v, v' = -k x - mu v."""
    import numpy as np
    from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator

    prob = harmonic_oscillator(k=1.0, mu=mu, u0=(1.0, 0.5))

    for u in [prob.u_init(), prob.u_exact(0.0)]:
        assert np.allclose([u.pos[0], u.vel[0]], prob.u0)

    t, h = 0.7, 1e-5
    u = prob.u_exact(t)
    up, um = prob.u_exact(t + h), prob.u_exact(t - h)
    assert np.isclose((up.pos[0] - um.pos[0]) / (2 * h), u.vel[0], atol=1e-8)
    assert np.isclose((up.vel[0] - um.vel[0]) / (2 * h), prob.eval_f(u, t)[0], atol=1e-8)
