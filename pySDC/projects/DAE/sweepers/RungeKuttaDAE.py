from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    RungeKutta,
    BackwardEuler,
    CrankNicholson,
    EDIRK4,
    DIRK43_2,
)


class RungeKuttaDAE(RungeKutta):
    r"""
    Custom sweeper class to implement Runge-Kutta (RK) methods for general differential-algebraic equations (DAEs)
    of the form

    .. math::
        0 = F(u, u', t).
    
    RK methods for general DAEs have the form

    .. math::
        0 = F(u_0 + \Delta t \sum_{j=1}^M a_{i,j} U_j, U_m),

    .. math::
        u_M = u_0 + \Delta t \sum_{j=1}^M b_j U_j.

    In pySDC, RK methods are implemented in the way that the coefficient matrix :math:`A` in the Butcher
    tableau is a lower triangular matrix so that the stages are updated node-by-node. This class therefore only supports
    RK methods with lower triangular matrices in the tableau.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    du_init : dtype_f
        Stores the initial condition for each step.

    Note
    ----
    When using a RK sweeper to simulate a problem make sure the DAE problem class has a ``du_exact`` method since RK methods need an initial
    condition for :math:`u'(t_0)` as well.

    In order to implement a new RK method for DAEs a new tableau can be added in ``pySDC.implementations.sweeper_classes.Runge_Kutta.py``.
    For example, a new method called ``newRungeKuttaMethod`` with nodes :math:`c=(c_1, c_2, c_3)`, weights :math:`b=(b_1, b_2, b_3)` and
    coefficient matrix

    ..math::
        \begin{eqnarray}
            A = \begin{pmatrix}
                a_{11} & 0 & 0 \\
                a_{21} & a_{22} & 0 \\
                a_{31} & a_{32} & & 0 \\
            \end{pmatrix}
        \end{eqnarray}

    can be implemented as follows:

    >>> class newRungeKuttaMethod(RungeKutta):
    >>>     nodes = np.array([c1, c2, c3])
    >>>     weights = np.array([b1, b2, b3])
    >>>     matrix = np.zeros((3, 3))
    >>>     matrix[0, 0] = a11
    >>>     matrix[1, :2] = [a21, a22]
    >>>     matrix[2, :] = [a31, a32, a33]
    >>>     ButcherTableauClass = ButcherTableau

    The new class ``newRungeKuttaMethodDAE`` can then be used by defining the DAE class inheriting from both, this base class and class containing
    the Butcher tableau:

    >>> class newRungeKuttaMethodDAE(RungeKuttaDAE, newRungeKuttaMethod):
    >>>     pass

    More details can be found [here](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/sweeper_classes/Runge_Kutta.py).
    """

    def __init__(self, params):
        super().__init__(params)
        self.du_init = None
        self.fully_initialized = False

    def predict(self):
        """
        Predictor to fill values with zeros at nodes before first sweep.
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        if not self.fully_initialized:
            self.du_init = prob.du_exact(lvl.time)
            self.fully_initialized = True

        lvl.f[0] = prob.dtype_f(self.du_init)
        for m in range(1, self.coll.num_nodes + 1):
            lvl.u[m] = prob.dtype_u(init=prob.init, val=0.0)
            lvl.f[m] = prob.dtype_f(init=prob.init, val=0.0)

        lvl.status.unlocked = True
        lvl.status.updated = True

    def integrate(self):
        r"""
        Returns the solution by integrating its gradient (fundamental theorem of calculus) at each collocation node.
        ``level.f`` stores the gradient of solution ``level.u``.

        Returns
        -------
        me : list of lists
            Integral of the gradient at each collocation node.
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        # integrate RHS over all collocation nodes
        me = []
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(prob.dtype_u(prob.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += lvl.dt * self.coll.Qmat[m, j] * lvl.f[j]

        return me

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        M = self.coll.num_nodes
        for m in range(M):
            u_approx = prob.dtype_u(lvl.u[0])
            for j in range(1, m + 1):
                u_approx += lvl.dt * self.QI[m + 1, j] * lvl.f[j][:]

            finit = lvl.f[m].flatten()
            lvl.f[m + 1][:] = prob.solve_system(
                fully_implicit_DAE.F,
                u_approx,
                lvl.dt * self.QI[m + 1, m + 1],
                finit,
                lvl.time + lvl.dt * self.coll.nodes[m + 1],
            )

        # Update numerical solution
        integral = self.integrate()
        for m in range(M):
            lvl.u[m + 1][:] = lvl.u[0][:] + integral[m][:]

        self.du_init = prob.dtype_f(lvl.f[-1])

        lvl.status.updated = True

        return None


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass


class TrapezoidalRuleDAE(RungeKuttaDAE, CrankNicholson):
    pass


class EDIRK4DAE(RungeKuttaDAE, EDIRK4):
    pass


class DIRK43_2DAE(RungeKuttaDAE, DIRK43_2):
    pass
