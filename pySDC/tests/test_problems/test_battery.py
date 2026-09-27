import pytest


@pytest.mark.base
@pytest.mark.parametrize('V_ref', [[-1.0, 1.0], [1.0, 2.0]])
def test_V_ref_is_checked(V_ref):
    import numpy as np
    from pySDC.implementations.problem_classes.Battery import battery_n_capacitors

    with pytest.raises(AssertionError, match='V_ref'):
        battery_n_capacitors(V_ref=np.array(V_ref), alpha=1.2)


@pytest.mark.base
def test_battery_implicit_inductance():
    """
    While the source supplies energy, the model is L i_L' = Vs - (Rs + R) i_L, and the implicit version has to agree
    with it for an inductance other than one.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.Battery import battery_implicit

    Vs, Rs, R, L = 5.0, 0.5, 1.0, 2.0
    prob = battery_implicit(Vs=Vs, Rs=Rs, R=R, L=L)
    rhs = prob.u_exact(0)
    rhs[:] = [0.3, 0.5]  # v_C < V_ref

    u = prob.solve_system(rhs, 0.1, rhs, 0)
    f = prob.eval_f(u, 0)
    assert np.allclose(u - 0.1 * f, rhs), 'solve_system does not invert eval_f'
    assert np.isclose(f[0], (Vs - (Rs + R) * u[0]) / L)
