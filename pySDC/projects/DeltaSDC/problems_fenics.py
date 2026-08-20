r"""
FEniCS problem with a node-local correction solve.

Kept in its own module because importing it requires ``dolfin``, which :mod:`problems` does not.

The correction problem is posed variationally, in the same shape as the stock ``solve_system``:
find :math:`\delta` such that

.. math::
    \langle \delta, q\rangle - factor\,\big[F(w+\delta; q) - F(w; q)\big] = \langle r, q\rangle .

The increment :math:`F(w+\delta) - F(w)` is **expanded analytically** rather than written as a
difference of two assembled forms. Assembling both and subtracting would cancel two
:math:`\mathcal{O}(|F|)` vectors and reinstate an absolute error of order
:math:`\varepsilon |F|`, which is exactly the failure mode the correction form exists to avoid.

For Gray-Scott, with :math:`w = (w_1, w_2)` the base and :math:`\delta = (\delta_1, \delta_2)`:

* diffusion is linear, so its increment is the same form evaluated at :math:`\delta`;
* :math:`A(1 - u_1) \to -A\delta_1` and :math:`B u_2 \to B\delta_2`;
* the reaction term expands as

  .. math::
      (w_1+\delta_1)(w_2+\delta_2)^2 - w_1 w_2^2
        = w_1\left(2 w_2 \delta_2 + \delta_2^2\right) + \delta_1 (w_2 + \delta_2)^2 ,

  in which every term carries an explicit factor :math:`\delta`.

Reduced precision is **emulated**: DOLFIN inherits PETSc's build-time scalar type, so values are
rounded through the requested working precision and written back, capping the *information* while
the arithmetic stays at the backend type. That is optimistic about iteration counts and attainable
accuracy compared with a real single-precision build.
"""

import dolfin as df
import numpy as np

from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott


def quantize_function(function, work_precision):
    """
    Round a DOLFIN function's coefficients through the working precision, in place.

    Parameters
    ----------
    function : dolfin.Function
        Function whose vector is quantized.
    work_precision : numpy.dtype or None
        Working precision. ``None`` leaves the function untouched.

    Returns
    -------
    dolfin.Function
        The same function, for convenience.
    """
    if work_precision is None:
        return function
    values = function.vector().get_local()
    function.vector().set_local(values.astype(np.dtype(work_precision)).astype(values.dtype))
    function.vector().apply('insert')
    return function


class fenics_grayscott_delta(fenics_grayscott):
    """
    Gray-Scott exposing ``solve_system_delta`` alongside the stock ``solve_system``.

    Parameters
    ----------
    solve_precision : dtype-like or None, optional
        Working precision to emulate for the node-local solve. ``None`` keeps backend precision.
    **kwargs
        Forwarded to :class:`fenics_grayscott`.
    """

    def __init__(self, solve_precision=None, **kwargs):
        """Initialization routine"""
        super().__init__(**kwargs)
        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)

        # base state of the correction, assigned per solve
        self.base = df.Function(self.V)
        self.delta = df.Function(self.V)

    def _increment_forms(self, test_functions):
        r"""
        Build the analytically expanded weak form of :math:`F(w+\delta) - F(w)`.

        Parameters
        ----------
        test_functions : tuple
            The two test functions of the mixed space.

        Returns
        -------
        ufl.Form
            The increment form, every term carrying an explicit factor of the correction.
        """
        q1, q2 = test_functions
        b1, b2 = df.split(self.base)
        d1, d2 = df.split(self.delta)

        # (b1+d1)(b2+d2)^2 - b1 b2^2, expanded so nothing cancels
        reaction = b1 * (2 * b2 * d2 + d2**2) + d1 * (b2 + d2) ** 2

        increment1 = (
            -self.Du * df.inner(df.nabla_grad(d1), df.nabla_grad(q1)) - reaction * q1 - self.A * d1 * q1
        ) * df.dx
        increment2 = (
            -self.Dv * df.inner(df.nabla_grad(d2), df.nabla_grad(q2)) + reaction * q2 - self.B * d2 * q2
        ) * df.dx
        return increment1 + increment2

    def solve_system_delta(self, r, factor, base, f_base, t):
        r"""
        Solve :math:`\delta - factor\,[f(base+\delta) - f(base)] = r` for the correction.

        Parameters
        ----------
        r : dtype_u
            Right-hand side of the correction equation.
        factor : float
            Implicit prefactor assembled by the sweeper.
        base : dtype_u
            Base state :math:`w`.
        f_base : dtype_f
            ``f`` evaluated at ``base``; accepted so no extra evaluation is needed.
        t : float
            Physical time, accepted for interface compatibility.

        Returns
        -------
        dtype_u
            The correction.
        """
        self.base.assign(base.values)
        self.delta.assign(df.Function(self.V))  # start from the zero correction

        q1, q2 = df.TestFunctions(self.V)
        d1, d2 = df.split(self.delta)
        r1, r2 = df.split(r.values)

        residual = (d1 * q1 + d2 * q2) * df.dx - factor * self._increment_forms((q1, q2))
        residual -= (r1 * q1 + r2 * q2) * df.dx

        trial = df.TrialFunction(self.V)
        jacobian = df.derivative(residual, self.delta, trial)

        problem = df.NonlinearVariationalProblem(residual, self.delta, [], jacobian)
        solver = df.NonlinearVariationalSolver(problem)
        prm = solver.parameters['newton_solver']
        prm['absolute_tolerance'] = 1e-09
        prm['relative_tolerance'] = 1e-08
        prm['maximum_iterations'] = 100
        prm['relaxation_parameter'] = 1.0
        solver.solve()

        quantize_function(self.delta, self.solve_precision)

        me = self.dtype_u(self.V)
        me.values.assign(self.delta)
        return me
