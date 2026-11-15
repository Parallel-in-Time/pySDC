import firedrake as fd

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake


class GrayScott1DFiredrake(FormFiredrake):
    r"""
    1D Gray-Scott reaction-diffusion system with Firedrake finite elements, given by its weak forms.

    The Gray-Scott system [1]_ describes two substances :math:`u` and :math:`v` that diffuse and react: :math:`u` is
    fed in at rate :math:`A` and used up by the reaction, which produces :math:`v`, and :math:`v` decays at rate
    :math:`B`,

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B v

    for :math:`x \in [0, 100]`, with zero-flux boundaries. Multiplying by test functions :math:`q_u, q_v` and
    integrating the Laplacians by parts gives the forms below; the boundary terms vanish, so there are no boundary
    conditions to impose. Everything else, the mass-matrix formulation, the solves and several levels, comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`. This is the same problem as
    :class:`~pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit.fenics_grayscott_mass`.

    Parameters
    ----------
    n : int, optional
        Number of cells, ignored if `mesh` is given.
    order : int, optional
        Degree of the Lagrange elements.
    variant : str, optional
        Where the Lagrange elements have their nodes. The default ``'equispaced'`` puts them where FEniCS does, so the
        interpolated initial value is the same; Firedrake's own default for high degrees is ``'spectral'``.
    Du, Dv : float, optional
        Diffusion rates of :math:`u` and :math:`v`.
    A : float, optional
        Feed rate of :math:`u`.
    B : float, optional
        Decay rate of :math:`v`.
    mesh : firedrake.MeshGeometry, optional
        A mesh of :math:`[0, 100]`, e.g. from a ``MeshHierarchy`` for several levels.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, see ``FormFiredrake``.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.

    References
    ----------
    .. [1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms
        of multistability. P. Gray, S. K. Scott. Chem. Eng. Sci. 38, 1 (1983).
    """

    def __init__(
        self,
        n=256,
        order=4,
        variant='equispaced',
        Du=1.0,
        Dv=0.01,
        A=0.09,
        B=0.086,
        mesh=None,
        solver_parameters=None,
        LHS_cache_size=12,
    ):
        """Initialization, see the class docstring for the arguments."""
        mesh = fd.IntervalMesh(n, 100) if mesh is None else mesh
        V = fd.FunctionSpace(mesh, 'CG', order, variant=variant)

        def mass(w, q):
            return fd.inner(w, q) * fd.dx

        def rhs(w, q):
            u, v = fd.split(w)
            q_u, q_v = fd.split(q)
            return (
                -Du * fd.inner(fd.grad(u), fd.grad(q_u))
                + (-u * v**2 + A * (1 - u)) * q_u
                - Dv * fd.inner(fd.grad(v), fd.grad(q_v))
                + (u * v**2 - B * v) * q_v
            ) * fd.dx

        x = fd.SpatialCoordinate(mesh)[0]
        bump = fd.sin(fd.pi * x / 100) ** 100
        super().__init__(
            V * V,
            mass,
            rhs,
            u_exact=fd.as_vector([1 - 0.5 * bump, 0.25 * bump]),
            solver_parameters=solver_parameters,
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('n', 'order', 'Du', 'Dv', 'A', 'B', localVars=locals(), readOnly=True)

    def u_exact(self, t):
        """
        The initial value. There is no exact solution for later times.

        Args:
            t (float): Time, must be 0

        Returns:
            dtype_u: The initial value
        """
        assert t == 0, 'There is no exact solution of Gray-Scott, only the initial value at t = 0'
        return super().u_exact(t)
