r"""
Discontinuous-Galerkin counterparts of the three CG examples, in mass-matrix form.

Each class is its CG parent with the weak form replaced: interior penalty for diffusion, Nitsche for
Dirichlet data, local Lax-Friedrichs for the Burgers advection. The Newton loop, mass matrix, initial
conditions and solver interface are inherited.

The ``penalty`` argument must be pinned to the fine level's value scaled by :math:`h_l / h_0`. The
SIPG form depends on the mesh and order it is built on, so rediscretising it per level changes the
penalty -- the dominant term -- and the coarse operator stops being :math:`P^T A_F P`. ``setups.py``
builds the pinned ladder.
"""

import dolfin as df

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott_mass
from pySDC.projects.FEM_with_FEniCS.problem_classes.Burgers_1D_FEniCS import fenics_burgers_mass


def sipg(u, v, kappa, mesh, alpha):
    r"""
    Symmetric interior penalty form for :math:`-\nabla \cdot (\kappa \nabla u)`, natural boundaries.

    The exterior facets carry no terms, so this alone imposes homogeneous Neumann conditions. Add
    :func:`nitsche` on top for Dirichlet data.

    Parameters
    ----------
    u : Function, TrialFunction or split component
        Argument of the operator. May be nonlinear in ``u``; ``df.derivative`` handles it.
    v : TestFunction or split component
        Test function.
    kappa : Constant or Expression
        Diffusion coefficient.
    mesh : Mesh
        Mesh the form lives on.
    alpha : float
        Penalty coefficient, entering as :math:`\alpha \kappa / h`. See the module docstring.

    Returns
    -------
    Form
        The bilinear form :math:`a(u, v)`, so a right-hand side uses ``-sipg(...)``.
    """

    n, h = df.FacetNormal(mesh), df.CellDiameter(mesh)
    alpha = df.Constant(alpha)

    return (
        kappa * df.inner(df.grad(u), df.grad(v)) * df.dx
        - kappa * df.inner(df.avg(df.grad(u)), df.jump(v, n)) * df.dS
        - kappa * df.inner(df.jump(u, n), df.avg(df.grad(v))) * df.dS
        + alpha * kappa / df.avg(h) * df.inner(df.jump(u, n), df.jump(v, n)) * df.dS
    )


def nitsche(u, v, u_D, kappa, mesh, alpha):
    r"""
    Weak (Nitsche) Dirichlet terms on the exterior boundary, to be added to :func:`sipg`.

    Returns the bilinear part, in :math:`u`, and the linear part carrying the boundary data
    :math:`u_D`. They are consistent by construction: for :math:`u = u_D` on the boundary the two
    data-dependent terms cancel.

    Parameters
    ----------
    u, v, kappa, mesh, alpha
        As in :func:`sipg`.
    u_D : Constant or Expression
        Boundary data.

    Returns
    -------
    a : Form
        Bilinear boundary terms, added to the operator.
    L : Form
        Linear boundary terms, a load vector.
    """

    n, h = df.FacetNormal(mesh), df.CellDiameter(mesh)
    alpha = df.Constant(alpha)

    a = (-kappa * df.dot(df.grad(u), n) * v - kappa * u * df.dot(df.grad(v), n) + alpha * kappa / h * u * v) * df.ds
    L = (-kappa * u_D * df.dot(df.grad(v), n) + alpha * kappa / h * u_D * v) * df.ds

    return a, L


class fenics_heat_dg_mass(fenics_heat_mass):
    r"""
    Forced 1D heat equation, DG in space, mass-matrix form.

    Same problem and same exact solution as :class:`fenics_heat_mass`; the Laplacian becomes SIPG
    and the Dirichlet condition :math:`u = c` is imposed by Nitsche terms.

    The inherited ``bc`` and ``bc_hom`` survive but have no dofs to act on in a DG space, so
    ``solve_system`` and the residual fix are inherited untouched and simply do nothing.

    Parameters
    ----------
    sigma : float, optional
        Interior/boundary penalty constant. Used only when ``penalty`` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`. A multilevel hierarchy has to pin this
        across levels, see the module docstring.
    family : str, optional
        Element family, ``'DG'``. Present so that the level hierarchy can pass it explicitly.
    **kwargs
        Forwarded to :class:`fenics_heat_mass`.

    Attributes
    ----------
    b_bc : GenericVector
        Assembled Nitsche load vector carrying the boundary data.
    """

    def __init__(self, sigma=10.0, penalty=None, family='DG', **kwargs):
        """Initialization routine"""

        super().__init__(family=family, **kwargs)
        self._makeAttributeAndRegister('sigma', 'penalty', localVars=locals(), readOnly=True)
        alpha = sigma * max(self.order, 1) ** 2 if penalty is None else penalty

        # no strong boundary conditions to fix the residual against
        self.fix_bc_for_residual = False

        mesh = self.V.mesh()
        u, v = df.TrialFunction(self.V), df.TestFunction(self.V)
        kappa = df.Constant(self.nu)
        a_bc, L_bc = nitsche(u, v, df.Constant(self.c), kappa, mesh, alpha)

        # K is the negated operator, matching the parent: M u' = K u + b_bc + M g
        self.K = df.assemble(-(sipg(u, v, kappa, mesh, alpha) + a_bc))
        self.b_bc = df.assemble(L_bc)

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side.

        The Nitsche load rides in the explicit part rather than in the implicit one, which keeps
        ``solve_system`` identical to the parent's. It is constant in time here, and every
        quadrature involved integrates constants exactly, so the split costs nothing: a constant
        does not enter the iteration matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f = super().eval_f(u, t)
        f.expl.values.vector().axpy(1.0, self.b_bc)

        return f


class fenics_burgers_dg_mass(fenics_burgers_mass):
    r"""
    Viscous Burgers in 1D, DG in space, mass-matrix form.

    The advection is taken in conservative form, :math:`(u^2/2)_x`, with a local Lax-Friedrichs
    numerical flux on the interior facets. The exterior facets use the interior trace, which leaves
    the boundary transmissive -- the same "no condition imposed" situation as the non-conservative
    CG form, and harmless for the cosine initial condition over the times run here.

    The viscous term is SIPG with natural boundaries.

    Parameters
    ----------
    sigma : float, optional
        Interior penalty constant. Used only when ``penalty`` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`. A multilevel hierarchy has to pin this
        across levels, see the module docstring.
    family : str, optional
        Element family, ``'DG'``.
    **kwargs
        Forwarded to :class:`fenics_burgers_mass`.
    """

    def __init__(self, sigma=10.0, penalty=None, family='DG', **kwargs):
        """Initialization routine"""

        super().__init__(family=family, **kwargs)
        self._makeAttributeAndRegister('sigma', 'penalty', localVars=locals(), readOnly=True)
        alpha = sigma * max(self.order, 1) ** 2 if penalty is None else penalty

        mesh = self.V.mesh()
        n = df.FacetNormal(mesh)
        v = df.TestFunction(self.V)
        w = self.w

        flux = df.as_vector([w**2 / 2])
        left, right = abs(w('+')), abs(w('-'))
        speed = df.conditional(df.gt(left, right), left, right)
        flux_hat = df.avg(flux) + 0.5 * speed * df.jump(w, n)

        self.F = (
            df.dot(flux, df.grad(v)) * df.dx
            - df.dot(flux_hat, df.jump(v, n)) * df.dS
            - df.dot(flux, n) * v * df.ds
            - sipg(w, v, df.Constant(self.nu), mesh, alpha)
        )


class fenics_grayscott_dg_mass(fenics_grayscott_mass):
    r"""
    Gray-Scott in 1D, DG in space, mass-matrix form.

    Both diffusion terms become SIPG with natural boundaries, matching the CG version's Neumann
    conditions. The reaction terms are cell-local and carry over verbatim.

    Parameters
    ----------
    sigma : float, optional
        Interior penalty constant. Used only when ``penalty`` is not given.
    penalty : float, optional
        Penalty coefficient, overriding :math:`\sigma p^2`. A multilevel hierarchy has to pin this
        across levels, see the module docstring.
    family : str, optional
        Element family, ``'DG'``.
    **kwargs
        Forwarded to :class:`fenics_grayscott_mass`.
    """

    def __init__(self, sigma=10.0, penalty=None, family='DG', **kwargs):
        """Initialization routine"""

        super().__init__(family=family, **kwargs)
        self._makeAttributeAndRegister('sigma', 'penalty', localVars=locals(), readOnly=True)
        alpha = sigma * max(self.order, 1) ** 2 if penalty is None else penalty

        mesh = self.V.mesh()
        q1, q2 = df.TestFunctions(self.V)
        w1, w2 = self.w1, self.w2

        self.F1 = (-w1 * (w2**2) * q1 + self.A * (1 - w1) * q1) * df.dx - sipg(
            w1, q1, df.Constant(self.Du), mesh, alpha
        )
        self.F2 = (w1 * (w2**2) * q2 - self.B * w2 * q2) * df.dx - sipg(w2, q2, df.Constant(self.Dv), mesh, alpha)
        self.F = self.F1 + self.F2
