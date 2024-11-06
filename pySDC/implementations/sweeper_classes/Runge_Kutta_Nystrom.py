import numpy as np
import logging

from pySDC.core.sweeper import Sweeper, _Pars
from pySDC.core.errors import ParameterError
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class ButcherTableauNoCollUpdate(object):
    """Version of Butcher Tableau that does not need a collocation update because the weights are put in the last line of Q"""

    def __init__(self, weights, nodes, matrix):
        """
        Initialization routine to get a quadrature matrix out of a Butcher tableau

        Args:
            weights (numpy.ndarray): Butcher tableau weights
            nodes (numpy.ndarray): Butcher tableau nodes
            matrix (numpy.ndarray): Butcher tableau entries
        """
        # check if the arguments have the correct form
        if type(matrix) != np.ndarray:
            raise ParameterError('Runge-Kutta matrix needs to be supplied as  a numpy array!')
        elif len(np.unique(matrix.shape)) != 1 or len(matrix.shape) != 2:
            raise ParameterError('Runge-Kutta matrix needs to be a square 2D numpy array!')

        if type(weights) != np.ndarray:
            raise ParameterError('Weights need to be supplied as a numpy array!')
        elif len(weights.shape) != 1:
            raise ParameterError(f'Incompatible dimension of weights! Need 1, got {len(weights.shape)}')
        elif len(weights) != matrix.shape[0]:
            raise ParameterError(f'Incompatible number of weights! Need {matrix.shape[0]}, got {len(weights)}')

        if type(nodes) != np.ndarray:
            raise ParameterError('Nodes need to be supplied as a numpy array!')
        elif len(nodes.shape) != 1:
            raise ParameterError(f'Incompatible dimension of nodes! Need 1, got {len(nodes.shape)}')
        elif len(nodes) != matrix.shape[0]:
            raise ParameterError(f'Incompatible number of nodes! Need {matrix.shape[0]}, got {len(nodes)}')

        self.globally_stiffly_accurate = np.allclose(matrix[-1], weights)

        self.tleft = 0.0
        self.tright = 1.0
        self.num_solution_stages = 0 if self.globally_stiffly_accurate else 1
        self.num_nodes = matrix.shape[0] + self.num_solution_stages
        self.weights = weights

        if self.globally_stiffly_accurate:
            # For globally stiffly accurate methods, the last row of the Butcher tableau is the same as the weights.
            self.nodes = np.append([0], nodes)
            self.Qmat = np.zeros([self.num_nodes + 1, self.num_nodes + 1])
            self.Qmat[1:, 1:] = matrix
        else:
            self.nodes = np.append(np.append([0], nodes), [1])
            self.Qmat = np.zeros([self.num_nodes + 1, self.num_nodes + 1])
            self.Qmat[1:-1, 1:-1] = matrix
            self.Qmat[-1, 1:-1] = weights  # this is for computing the solution to the step from the previous stages

        self.left_is_node = True
        self.right_is_node = self.nodes[-1] == self.tright

        # compute distances between the nodes
        if self.num_nodes > 1:
            self.delta_m = self.nodes[1:] - self.nodes[:-1]
        else:
            self.delta_m = np.zeros(1)
        self.delta_m[0] = self.nodes[0] - self.tleft

        # check if the RK scheme is implicit
        self.implicit = any(matrix[i, i] != 0 for i in range(self.num_nodes - self.num_solution_stages))


class RungeKuttaNystrom(RungeKutta):
    """
    Runge-Kutta scheme that fits the interface of a sweeper.
    Actually, the sweeper idea fits the Runge-Kutta idea when using only lower triangular rules, where solutions
    at the nodes are successively computed from earlier nodes. However, we only perform a single iteration of this.

    We have two choices to realise a Runge-Kutta sweeper: We can choose Q = Q_Delta = <Butcher tableau>, but in this
    implementation, that would lead to a lot of wasted FLOPS from integrating with Q and then with Q_Delta and
    subtracting the two. For that reason, we built this new sweeper, which does not have a preconditioner.

    This class only supports lower triangular Butcher tableaux such that the system can be solved with forward
    substitution. In this way, we don't get the maximum order that we could for the number of stages, but computing the
    stages is much cheaper. In particular, if the Butcher tableaux is strictly lower triangular, we get an explicit
    method, which does not require us to solve a system of equations to compute the stages.

    Please be aware that all fundamental parameters of the Sweeper are ignored. These include

     - num_nodes
     - collocation_class
     - initial_guess
     - QI

    All of these variables are either determined by the RK rule, or are not part of an RK scheme.

    Attribues:
        butcher_tableau (ButcherTableauNoCollUpdate): Butcher tableau for the Runge-Kutta scheme that you want
    """

    ButcherTableauClass = ButcherTableauNoCollUpdate
    weights_bar = None
    matrix_bar = None

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        super().__init__(params)
        self.coll_bar = self.get_Butcher_tableau_bar()
        self.Qx = self.coll_bar.Qmat

    @classmethod
    def get_Butcher_tableau_bar(cls):
        return cls.ButcherTableauClass(cls.weights_bar, cls.nodes, cls.matrix_bar)

    def get_full_f(self, f):
        """
        Test the right hand side function is the correct type

        Args:
            f (dtype_f): Right hand side at a single node

        Returns:
            mesh: Full right hand side as a mesh
        """

        if type(f) in [particles, fields, acceleration]:
            return f
        else:
            raise NotImplementedError(f'Type \"{type(f)}\" not implemented')

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(L.u[0])
            rhs.pos += L.dt * self.coll.nodes[m + 1] * L.u[0].vel

            for j in range(1, m + 1):
                # build RHS from f-terms (containing the E field) and the B field

                f = P.build_f(L.f[j], L.u[j], L.time + L.dt * self.coll.nodes[j])
                rhs.pos += L.dt**2 * self.Qx[m + 1, j] * self.get_full_f(f)
                """

                Implicit part only works for Velocity-Verlet scheme
                Boris solver for the implicit part

                """

                if self.coll.implicit:
                    ck = rhs.vel * 0.0
                    L.f[3] = P.eval_f(rhs, L.time + L.dt)
                    rhs.vel = P.boris_solver(ck, L.dt, L.f[0], L.f[3], L.u[0])

                else:
                    rhs.vel += L.dt * self.QI[m + 1, j] * self.get_full_f(f)

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m + 1] = rhs
            # update function values
            if self.coll.implicit:
                # That is why it only works for the Velocity-Verlet scheme
                L.f[0] = P.eval_f(L.u[0], L.time)
                L.f[m + 1] = P.dtype_f(L.f[0])
            else:
                if m != self.coll.num_nodes - 1:
                    L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level

        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        In this Runge-Kutta implementation, the solution to the step is always stored in the last node
        """
        self.level.uend = self.level.u[-1]


class RKN(RungeKuttaNystrom):
    """
    Runge-Kutta-Nystrom method
    https://link.springer.com/book/10.1007/978-3-540-78862-1
    page: 284
    Chapter: II.14 Numerical methods for Second order differential equations
    """

    nodes = np.array([0.0, 0.5, 0.5, 1])
    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    matrix = np.zeros([4, 4])
    matrix[1, 0] = 0.5
    matrix[2, 1] = 0.5
    matrix[3, 2] = 1.0

    weights_bar = np.array([1.0, 1.0, 1.0, 0]) / 6.0
    matrix_bar = np.zeros([4, 4])
    matrix_bar[1, 0] = 1 / 8
    matrix_bar[2, 0] = 1 / 8
    matrix_bar[3, 2] = 1 / 2


class Velocity_Verlet(RungeKuttaNystrom):
    """
    Velocity-Verlet scheme
    https://de.wikipedia.org/wiki/Verlet-Algorithmus
    """

    nodes = np.array([1.0, 1.0])
    weights = np.array([1 / 2, 0])
    matrix = np.zeros([2, 2])
    matrix[1, 1] = 1
    weights_bar = np.array([1 / 2, 0])
    matrix_bar = np.zeros([2, 2])
