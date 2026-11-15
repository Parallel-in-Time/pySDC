import firedrake as fd

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX


class ConvDiff2DFiredrake(FormFiredrakeIMEX):
    r"""
    2D convection-diffusion of a rotating Gaussian on :math:`[-0.5, 0.5]^2` with time-dependent Dirichlet values,
    IMEX, with Firedrake and given by its weak forms. This is the port of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.ConvectionDiffusion_2D_FEniCS.fenics_ConvDiff2D_mass`, on
    the same mesh.

    The equation :math:`u_t = -U \cdot \nabla u + \nu \Delta u` with the rotation :math:`U = (-4y, 4x)` has the exact
    solution

    .. math::
        u = \frac{\sigma^2}{\sigma^2 + 4 \nu t}
        \exp\left(-\frac{(\hat{x} - x_0)^2 + (\hat{y} - y_0)^2}{\sigma^2 + 4 \nu t}\right),

    with the rotated coordinates :math:`\hat{x} = \cos(4t) x + \sin(4t) y`, :math:`\hat{y} = -\sin(4t) x + \cos(4t) y`
    and the initial centre :math:`(x_0, y_0) = (-0.25, 0)`. The diffusion is treated implicitly, the convection
    explicitly. Everything else comes from
    :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrakeIMEX`.

    Parameters
    ----------
    c_nvars : int, optional
        Number of squares per direction, each split into two triangles.
    order : int, optional
        Degree of the Lagrange elements.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    sigma : float, optional
        Initial width :math:`\sigma` of the Gaussian.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, c_nvars=64, order=2, nu=0.01, sigma=0.05, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        # Firedrake takes the upper corner here, not the lengths; `diagonal='right'` and equispaced nodes are DOLFIN's
        mesh = fd.RectangleMesh(c_nvars, c_nvars, 0.5, 0.5, originX=-0.5, originY=-0.5, diagonal='right')
        V = fd.FunctionSpace(mesh, 'CG', order, variant='equispaced')

        t = fd.Constant(0.0)
        x, y = fd.SpatialCoordinate(mesh)
        x_hat = fd.cos(4 * t) * x + fd.sin(4 * t) * y
        y_hat = -fd.sin(4 * t) * x + fd.cos(4 * t) * y
        width = sigma**2 + 4 * nu * t
        u_exact = sigma**2 / width * fd.exp(-((x_hat + 0.25) ** 2 + y_hat**2) / width)
        velocity = fd.as_vector([-4 * y, 4 * x])

        def mass(u, v):
            return u * v * fd.dx

        def diffusion(u, v):
            return -nu * fd.inner(fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx

        def convection(u, v):
            return -fd.inner(fd.dot(velocity, fd.nabla_grad(u)), v) * fd.dx

        super().__init__(
            V,
            mass,
            diffusion,
            convection,
            bcs=(fd.DirichletBC(V, u_exact, 'on_boundary'),),
            t=t,
            u_exact=u_exact,
            # the implicit systems are linear
            solver_parameters={'snes_type': 'ksponly'},
            LHS_cache_size=LHS_cache_size,
        )
        self._makeAttributeAndRegister('c_nvars', 'order', 'nu', 'sigma', localVars=locals(), readOnly=True)
        self.mesh = mesh
