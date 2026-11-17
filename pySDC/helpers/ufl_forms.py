r"""
Form manipulation for problems given as UFL weak forms, written against ``ufl`` only.

A problem is given by a mass form :math:`m(u, v)` and right-hand side forms :math:`r(u, v)` (and optionally an explicit
part :math:`r_E(u, v)`), all as callables of the solution :math:`u` and the test function :math:`v`. The semi-discrete
system is :math:`M u' = R(u)`. Nothing here imports a finite element backend: the backends (for now only
:class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`) do the assembly and the solves, and
another UFL-based backend could reuse this module as it is.
"""

import ufl


def total_rhs(rhs, rhs_exp=None):
    r"""
    Join the implicit and the explicit right-hand side into one, for sweepers that treat everything implicitly.

    Args:
        rhs (callable): Implicit part :math:`r(u, v)`
        rhs_exp (callable, optional): Explicit part :math:`r_E(u, v)`

    Returns:
        callable: :math:`(u, v) \mapsto r(u, v) + r_E(u, v)`, or `rhs` itself if there is no explicit part
    """
    if rhs_exp is None:
        return rhs
    return lambda u, v: rhs(u, v) + rhs_exp(u, v)


def implicit_system(mass, rhs, u, v, du, factor):
    r"""
    Residual and Jacobian of the implicit Euler-like system :math:`M u - \text{factor}\, R(u) = b`, without :math:`b`.

    Args:
        mass (callable): Mass form :math:`m(u, v)`
        rhs (callable): Right-hand side :math:`r(u, v)` that is treated implicitly
        u (ufl.Coefficient): The unknown
        v (ufl.Argument): Test function
        du (ufl.Argument): Trial function for the Jacobian. Backends need an argument of their own type here, which
            is why it is not made from `u`.
        factor: Scalar UFL expression multiplying the right-hand side, e.g. a backend constant. Using a constant
            rather than a float keeps the form signature the same for all factors, so the backend compiles once.

    Returns:
        tuple: Residual form :math:`F(u; v)` and its Jacobian :math:`\partial F / \partial u` in direction `du`
    """
    F = mass(u, v) - factor * rhs(u, v)
    return F, ufl.derivative(F, u, du)


def time_derivative(expr, t):
    r"""
    Differentiate a UFL expression with respect to the time `t` it depends on.

    UFL differentiates only with respect to variables, so `t` is replaced by a variable wrapping it first.

    Args:
        expr (ufl.core.expr.Expr): Expression depending on `t`
        t (ufl.core.terminal.Terminal): The time, e.g. a backend constant

    Returns:
        ufl.core.expr.Expr: :math:`\partial \text{expr} / \partial t`
    """
    t_var = ufl.variable(t)
    return ufl.diff(ufl.replace(expr, {t: t_var}), t_var)


def manufactured_forcing(mass, rhs, u_exact, t, v):
    r"""
    Forcing that makes `u_exact` solve :math:`M u' = R(u) + g`, i.e. :math:`g(v) = m(\partial_t u_e, v) - r(u_e, v)`.

    Add it to a right-hand side form to get a problem with known solution. The time derivative is taken by UFL, so the
    forcing is exact for whatever `u_exact` is given.

    Args:
        mass (callable): Mass form :math:`m(u, v)`
        rhs (callable): Full right-hand side :math:`r(u, v)`, see :func:`total_rhs` for joining implicit and explicit
            parts
        u_exact (ufl.core.expr.Expr): Exact solution, depending on space through a spatial coordinate and on `t`
        t (ufl.core.terminal.Terminal): The time `u_exact` depends on
        v (ufl.Argument): Test function

    Returns:
        ufl.Form: The forcing :math:`g(v)`, which does not depend on the solution
    """
    return mass(time_derivative(u_exact, t), v) - rhs(u_exact, v)
