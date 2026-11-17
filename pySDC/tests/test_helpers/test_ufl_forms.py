"""
Tests for the backend-free form helpers. They use UFL alone, without Firedrake's elements or assembly, so they also
check that the helpers stay backend-free. UFL is not in the base env, so they run with the Firedrake tests.

UFL has no finite elements of its own, so a minimal scalar Lagrange element stands in for a backend's. The forms are
checked by evaluating their integrands at a point, which UFL can do without assembling anything.
"""

import pytest


def setup():
    """Domain, function space, coefficient, arguments and time constant on an interval, built with UFL alone."""
    import ufl
    from ufl.finiteelement import AbstractFiniteElement
    from ufl.pullback import identity_pullback
    from ufl.sobolevspace import H1

    class P1(AbstractFiniteElement):
        """Scalar Lagrange element of degree 1, just enough for UFL to build forms."""

        def __repr__(self):
            return 'P1()'

        def __str__(self):
            return 'P1'

        def __hash__(self):
            return hash(repr(self))

        def __eq__(self, other):
            return isinstance(other, P1)

        sobolev_space = H1
        pullback = identity_pullback
        embedded_superdegree = 1
        embedded_subdegree = 1
        cell = ufl.interval
        reference_value_shape = ()
        sub_elements = []

        def _ufl_hash_data_(self):
            return repr(self)

        def _ufl_signature_data_(self):
            return repr(self)

    class CoordinateElement(P1):
        """The same element, vector valued, for the mesh coordinates."""

        def __repr__(self):
            return 'CoordinateElement()'

        reference_value_shape = (1,)

    domain = ufl.Mesh(CoordinateElement())
    V = ufl.FunctionSpace(domain, P1())
    return {
        'u': ufl.Coefficient(V),
        'v': ufl.TestFunction(V),
        'du': ufl.TrialFunction(V),
        't': ufl.Constant(domain),
        'x': ufl.SpatialCoordinate(domain)[0],
    }


def integrand_at(form, mapping, x=0.3):
    """Evaluate the integrands of a form over one cell at the point `x`, with terminals replaced by `mapping`."""
    from ufl.algorithms import expand_derivatives

    return sum(integral.integrand()((x,), mapping) for integral in expand_derivatives(form).integrals())


@pytest.mark.firedrake
def test_total_rhs():
    import ufl
    from pySDC.helpers.ufl_forms import total_rhs

    s = setup()

    def rhs(u, v):
        return -(u**3) * v * ufl.dx

    def rhs_exp(u, v):
        return 2 * u * v * ufl.dx

    assert total_rhs(rhs) is rhs
    full = total_rhs(rhs, rhs_exp)(s['u'], s['v'])
    assert integrand_at(full, {s['u']: 2.0, s['v']: 1.0}) == pytest.approx(-8 + 4)


@pytest.mark.firedrake
def test_implicit_system():
    """F(u) = m(u, v) - factor r(u, v) and its Jacobian, for a nonlinear right-hand side."""
    import ufl
    from pySDC.helpers.ufl_forms import implicit_system

    s = setup()

    def mass(u, v):
        return u * v * ufl.dx

    def rhs(u, v):
        return -(u**3) * v * ufl.dx

    factor = 0.5

    F, J = implicit_system(mass, rhs, s['u'], s['v'], s['du'], factor)
    assert len(F.arguments()) == 1 and len(J.arguments()) == 2

    point = {s['u']: 2.0, s['v']: 1.0, s['du']: 1.0}
    assert integrand_at(F, point) == pytest.approx(2 - factor * (-8))
    assert integrand_at(J, point) == pytest.approx(1 - factor * (-3 * 2.0**2))


@pytest.mark.firedrake
def test_time_derivative():
    import ufl
    from pySDC.helpers.ufl_forms import time_derivative

    s = setup()
    t, x = s['t'], s['x']
    expr = ufl.cos(t) * x + t**2

    dt_expr = ufl.algorithms.expand_derivatives(time_derivative(expr, t))
    assert dt_expr((0.3,), {t: 0.7}) == pytest.approx(-ufl.sin(0.7) * 0.3 + 2 * 0.7)


@pytest.mark.firedrake
def test_manufactured_forcing():
    """g(v) = m(u_e', v) - r(u_e, v), with the time derivative from UFL."""
    import numpy as np
    import ufl
    from pySDC.helpers.ufl_forms import manufactured_forcing

    s = setup()
    t, x, v = s['t'], s['x'], s['v']

    def mass(u, v):
        return u * v * ufl.dx

    def rhs(u, v):
        return -(u**3) * v * ufl.dx

    u_exact = ufl.cos(t) * x

    g = manufactured_forcing(mass, rhs, u_exact, t, v)
    assert g.arguments() == (v,), 'the forcing must not depend on the solution'

    tt, xx = 0.7, 0.3
    expect = -np.sin(tt) * xx + (np.cos(tt) * xx) ** 3
    assert integrand_at(g, {t: tt, v: 1.0}, x=xx) == pytest.approx(expect)
