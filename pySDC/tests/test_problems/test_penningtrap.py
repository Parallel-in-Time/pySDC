import numpy as np
import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [1, 2, 9])
def test_interactions_match_pairwise_sum(N):
    """The all-pairs interaction against the plain double loop over particles, with charges that differ."""
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap

    rng = np.random.default_rng(0)
    pos, q, sig = rng.standard_normal((3, N)), rng.uniform(0.5, 2, N), 0.1

    expected = np.zeros((3, N))
    for i in range(N):
        for j in range(N):
            d = pos[:, i] - pos[:, j]
            expected[:, i] += q[j] * d / (d @ d + sig**2) ** 1.5

    # the sums over particles run in another order, so they agree to rounding, not bit for bit
    assert np.allclose(penningtrap.fast_interactions(N, pos, sig, q), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.base
def test_vectorised_particle_updates_match_per_particle_loops():
    """
    The right-hand side, the fields and the Boris solve act on all particles at once; each must match the
    per-particle formula bit for bit, except the Boris velocity, whose rotation takes a norm that may sum its three
    squares in another order. Charges and masses differ per particle here, unlike in ``u_init``, so a broadcast
    along the wrong axis cannot hide behind q = m = 1.
    """
    from pySDC.implementations.datatype_classes.particles import particles
    from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap

    rng = np.random.default_rng(0)
    N = 7
    P = penningtrap(
        omega_B=25.0, omega_E=4.9, u0=np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object), nparts=N, sig=0.1
    )
    u = P.u_init()
    u.pos[:], u.vel[:] = rng.standard_normal((3, N)), rng.standard_normal((3, N))
    u.q[:], u.m[:] = rng.uniform(0.5, 2, N), rng.uniform(0.5, 2, N)

    f = P.eval_f(u, 0.0)
    Emat = np.diag([1, 1, -2])
    interactions = P.get_interactions(u)
    for n in range(N):
        elec = interactions[:, n] + P.omega_E**2 / (u.q[n] / u.m[n]) * np.dot(Emat, u.pos[:, n])
        assert np.array_equal(f.elec[:, n], elec)
        assert np.array_equal(f.magn[:, n], P.omega_B * np.array([0, 0, 1]))

    rhs = P.build_f(f, u, 0.0)
    for n in range(N):
        assert np.array_equal(rhs[:, n], u.q[n] / u.m[n] * (f.elec[:, n] + np.cross(u.vel[:, n], f.magn[:, n])))

    new_fields = P.eval_f(u, 0.1)
    new_fields.magn[:] = rng.standard_normal((3, N))  # a magnetic field that changes, so the c-term update is not zero
    c = particles.velocity(P.init)
    c[:] = rng.standard_normal((3, N))
    c_before, dt = c.copy(), 0.05
    vel = P.boris_solver(c, dt, f, new_fields, u)
    Emean = 0.5 * (f.elec + new_fields.elec)
    for n in range(N):
        a = u.q[n] / u.m[n]
        cn = c_before[:, n] + dt / 2 * a * np.cross(u.vel[:, n], f.magn[:, n] - new_fields.magn[:, n])
        assert np.array_equal(c[:, n], cn), 'the c term must still be updated in the caller\'s array'
        vm = u.vel[:, n] + dt / 2 * a * Emean[:, n] + cn / 2
        t = dt / 2 * a * new_fields.magn[:, n]
        s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
        vp = vm + np.cross(vm + np.cross(vm, t), s)
        assert np.allclose(vel[:, n], vp + dt / 2 * a * Emean[:, n] + cn / 2, rtol=1e-15, atol=0)
