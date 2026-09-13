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
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat


def quantize_function(function, work_precision, normalize=False):
    r"""
    Round a DOLFIN function's coefficients through the working precision, in place.

    Parameters
    ----------
    function : dolfin.Function
        Function whose vector is quantized.
    work_precision : numpy.dtype or None
        Working precision. ``None`` leaves the function untouched.
    normalize : bool, optional
        Scale the coefficients to :math:`\mathcal{O}(1)` before rounding and scale them back after.
        A *correction* has to be quantized this way in half precision: the smallest ``float16``
        subnormal is 6e-8, so a correction of that size or below rounds to zero and the solver
        returns nothing. The delta form makes the solver's argument small, which is exactly what
        half precision cannot represent, so the two are in tension unless the unknown is scaled.

    Returns
    -------
    dolfin.Function
        The same function, for convenience.
    """
    if work_precision is None:
        return function
    values = function.vector().get_local()
    scale = max(float(np.max(np.abs(values))), 1e-300) if normalize else 1.0
    rounded = (values / scale).astype(np.dtype(work_precision)).astype(values.dtype) * scale
    function.vector().set_local(rounded)
    function.vector().apply('insert')
    return function


class fenics_grayscott_delta(fenics_grayscott):
    r"""
    Gray-Scott exposing ``solve_system_delta`` alongside the stock ``solve_system``.

    Parameters
    ----------
    solve_precision : dtype-like or None, optional
        Working precision to emulate for the node-local solve. ``None`` keeps backend precision.
    normalize : bool, optional
        Scale the correction to :math:`\mathcal{O}(1)` around the quantization. Half precision
        needs it; see :func:`quantize_function`.
    **kwargs
        Forwarded to :class:`fenics_grayscott`.
    """

    def __init__(self, solve_precision=None, normalize=True, **kwargs):
        """Initialization routine"""
        # Dolfin's Newton stops on |r_k| < newton_tol OR |r_k|/|r_0| < newton_rtol, and here the
        # unknown is the correction, started from zero -- so r_0 *is* the correction right-hand
        # side, which shrinks as the SDC iteration converges. The relative bar is therefore already
        # an adaptive one, tightening by itself sweep after sweep, and the only thing that can spoil
        # it is a fixed absolute bar firing first. So the absolute bar is off by default here, unlike
        # in the parent, whose unknown is the full state and whose r_0 does not shrink.
        kwargs.setdefault('newton_tol', 1e-30)
        super().__init__(**kwargs)
        self.solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        self.normalize = normalize

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

    def eval_f_increment(self, base, delta, t):
        r"""
        Evaluate :math:`f(w+\delta) - f(w)` from the analytically expanded weak form.

        Same form :meth:`solve_system_delta` uses inside the solve, assembled and multiplied by the
        inverse mass matrix so it matches what ``eval_f`` returns. Assembling
        :math:`F(w+\delta)` and :math:`F(w)` separately and subtracting would cancel two
        :math:`\mathcal{O}(|F|)` vectors, which is what this exists to avoid.

        Parameters
        ----------
        base : dtype_u
            The base state :math:`w`.
        delta : dtype_u
            The correction :math:`\delta`.
        t : float
            Physical time, accepted for interface compatibility.

        Returns
        -------
        dtype_f
            The increment.
        """
        self.base.assign(base.values)
        self.delta.assign(delta.values)

        assembled = self.dtype_f(self.V)
        assembled.values = df.Function(self.V, df.assemble(self._increment_forms(df.TestFunctions(self.V))))

        me = self.dtype_f(self.V)
        df.solve(1.0 * self.M, me.values.vector(), assembled.values.vector())
        return me

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
        prm['absolute_tolerance'] = self.newton_tol
        prm['relative_tolerance'] = self.newton_rtol
        prm['maximum_iterations'] = self.newton_maxiter
        prm['relaxation_parameter'] = 1.0
        solver.solve()

        quantize_function(self.delta, self.solve_precision, self.normalize)

        me = self.dtype_u(self.V)
        me.values.assign(self.delta)
        return me


class fenics_heat_no_increment(fenics_heat):
    r"""
    Control: the heat equation with its analytic increment deliberately out of reach.

    ``fenics_heat`` supplies ``eval_f_increment``, so a sweeper on it never falls back to forming
    :math:`\Delta f` by subtracting two stored right-hand sides. This class hides it again, which is
    what makes the claim that the increment matters falsifiable -- on this problem
    :math:`|M^{-1}Ku|` is of order 1e5, so the subtraction is already 7.7e-11 off in double
    precision and considerably worse below it.

    The attribute raises rather than being absent, because the sweeper dispatches on ``hasattr`` and
    a raising property is the way to make that report ``False`` for an inherited method.
    """

    @property
    def eval_f_increment(self):
        """
        Raises
        ------
        AttributeError
            Always. That is the point of the class.
        """
        raise AttributeError('control: the analytic increment is deliberately unavailable here')
