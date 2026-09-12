import dolfin as df


class NewtonStep(df.NonlinearProblem):
    r"""
    Newton problem for a single SDC node-to-node step :math:`M w + \Delta t_{QI} N(w) = rhs`.

    The right-hand side handed over by the sweeper is an assembled vector, and it is subtracted
    from the residual here, at the algebraic level. The alternative -- writing it into the
    variational form as :math:`\int_\Omega rhs \cdot v\,dx` so that dolfin's high level
    ``solve(F == 0, ...)`` interface can be used -- applies the mass matrix to it, which then
    has to be undone by a mass matrix solve beforehand. That round trip is exact in exact
    arithmetic, so it buys nothing, while costing a solve per node per sweep and capping the
    attainable accuracy at the tolerance of that solve.

    ``rhs`` and ``bcs`` are set per solve rather than at construction, so that the form and its
    Jacobian can be compiled once and reused with the step size carried by a ``Constant``.

    Parameters
    ----------
    F : UFL form
        Residual form of the step, *without* the right-hand side term.
    J : UFL form
        Jacobian of ``F``.

    Attributes
    ----------
    rhs : GenericVector
        Right-hand side vector for the current solve, subtracted from the residual.
    bcs : list of DirichletBC
        Boundary conditions for the current solve, applied in residual form.
    """

    def __init__(self, F, J):
        super().__init__()
        self.F_form = F
        self.J_form = J
        self.rhs = None
        self.bcs = []

    def F(self, b, x):
        df.assemble(self.F_form, tensor=b)
        b.axpy(-1.0, self.rhs)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        df.assemble(self.J_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
