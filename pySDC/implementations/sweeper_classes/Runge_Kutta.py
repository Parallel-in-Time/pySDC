import numpy as np
import logging

from pySDC.core.Sweeper import sweeper, _Pars
from pySDC.core.Errors import ParameterError
from pySDC.core.Level import level
from pySDC.implementations.datatype_classes.mesh import imex_mesh, mesh


class ButcherTableau(object):
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

        # Set number of nodes, left and right interval boundaries
        self.num_solution_stages = 1
        self.num_nodes = matrix.shape[0] + self.num_solution_stages
        self.tleft = 0.0
        self.tright = 1.0

        self.nodes = np.append(np.append([0], nodes), [1])
        self.weights = weights
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


class ButcherTableauEmbedded(object):
    def __init__(self, weights, nodes, matrix):
        """
        Initialization routine to get a quadrature matrix out of a Butcher tableau for embedded RK methods.

        Be aware that the method that generates the final solution should be in the first row of the weights matrix.

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
        elif len(weights.shape) != 2:
            raise ParameterError(f'Incompatible dimension of weights! Need 2, got {len(weights.shape)}')
        elif len(weights[0]) != matrix.shape[0]:
            raise ParameterError(f'Incompatible number of weights! Need {matrix.shape[0]}, got {len(weights[0])}')

        if type(nodes) != np.ndarray:
            raise ParameterError('Nodes need to be supplied as a numpy array!')
        elif len(nodes.shape) != 1:
            raise ParameterError(f'Incompatible dimension of nodes! Need 1, got {len(nodes.shape)}')
        elif len(nodes) != matrix.shape[0]:
            raise ParameterError(f'Incompatible number of nodes! Need {matrix.shape[0]}, got {len(nodes)}')

        # Set number of nodes, left and right interval boundaries
        self.num_solution_stages = 2
        self.num_nodes = matrix.shape[0] + self.num_solution_stages
        self.tleft = 0.0
        self.tright = 1.0

        self.nodes = np.append(np.append([0], nodes), [1, 1])
        self.weights = weights
        self.Qmat = np.zeros([self.num_nodes + 1, self.num_nodes + 1])
        self.Qmat[1:-2, 1:-2] = matrix
        self.Qmat[-1, 1:-2] = weights[0]  # this is for computing the higher order solution
        self.Qmat[-2, 1:-2] = weights[1]  # this is for computing the lower order solution

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


class RungeKutta(sweeper):
    nodes = None
    weights = None
    matrix = None
    ButcherTableauClass = ButcherTableau

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

    The entries of the Butcher tableau are stored as class attributes.
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        # set up logger
        self.logger = logging.getLogger('sweeper')

        # check if some parameters are set which only apply to actual sweepers
        for key in ['initial_guess', 'collocation_class', 'num_nodes']:
            if key in params:
                self.logger.warning(f'"{key}" will be ignored by Runge-Kutta sweeper')

        # set parameters to their actual values
        self.coll = self.get_Butcher_tableau()
        params['initial_guess'] = 'zero'
        params['collocation_class'] = type(self.ButcherTableauClass)
        params['num_nodes'] = self.coll.num_nodes

        # disable residual computation by default
        params['skip_residual_computation'] = params.get(
            'skip_residual_computation', ('IT_CHECK', 'IT_FINE', 'IT_COARSE', 'IT_UP', 'IT_DOWN')
        )

        # check if we can skip some usually unnecessary right hand side evaluations
        params['eval_rhs_at_right_boundary'] = params.get('eval_rhs_at_right_boundary', False)

        self.params = _Pars(params)

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        self.parallelizable = False
        self.QI = self.coll.Qmat

    @classmethod
    def get_Q_matrix(cls):
        return cls.get_Butcher_tableau().Qmat

    @classmethod
    def get_Butcher_tableau(cls):
        return cls.ButcherTableauClass(cls.weights, cls.nodes, cls.matrix)

    @classmethod
    def get_update_order(cls):
        """
        Get the order of the lower order method for doing adaptivity. Only applies to embedded methods.
        """
        raise NotImplementedError(
            f"There is not an update order for RK scheme \"{cls.__name__}\" implemented. Maybe it is not an embedded scheme?"
        )

    def get_full_f(self, f):
        """
        Get the full right hand side as a `mesh` from the right hand side

        Args:
            f (dtype_f): Right hand side at a single node

        Returns:
            mesh: Full right hand side as a mesh
        """
        if type(f) == mesh:
            return f
        elif type(f) == imex_mesh:
            return f.impl + f.expl
        elif f is None:
            prob = self.level.prob
            return self.get_full_f(prob.dtype_f(prob.init, val=0))
        else:
            raise NotImplementedError(f'Type \"{type(f)}\" not implemented in Runge-Kutta sweeper')

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(prob.dtype_u(prob.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += lvl.dt * self.coll.Qmat[m, j] * self.get_full_f(lvl.f[j])

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes

        Returns:
            None
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = lvl.u[0]
            for j in range(1, m + 1):
                rhs += lvl.dt * self.QI[m + 1, j] * self.get_full_f(lvl.f[j])

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            if self.coll.implicit:
                lvl.u[m + 1][:] = prob.solve_system(
                    rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m]
                )
            else:
                lvl.u[m + 1][:] = rhs[:]

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            if m < M - self.coll.num_solution_stages or self.params.eval_rhs_at_right_boundary:
                lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None

    def compute_end_point(self):
        """
        In this Runge-Kutta implementation, the solution to the step is always stored in the last node
        """
        self.level.uend = self.level.u[-1]

    @property
    def level(self):
        """
        Returns the current level

        Returns:
            pySDC.Level.level: Current level
        """
        return self.__level

    @level.setter
    def level(self, lvl):
        """
        Sets a reference to the current level (done in the initialization of the level)

        Args:
            lvl (pySDC.Level.level): Current level
        """
        assert isinstance(lvl, level), f"You tried to set the sweeper's level with an instance of {type(lvl)}!"
        if lvl.params.restol > 0:
            lvl.params.restol = -1
            self.logger.warning(
                'Overwriting residual tolerance with -1 because RK methods are direct and hence may not compute a residual at all!'
            )

        self.__level = lvl

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        for m in range(1, self.coll.num_nodes + 1):
            lvl.u[m] = prob.dtype_u(init=prob.init, val=0.0)

        # indicate that this level is now ready for sweeps
        lvl.status.unlocked = True
        lvl.status.updated = True


class ForwardEuler(RungeKutta):
    """
    Forward Euler. Still a classic.

    Not very stable first order method.
    """

    nodes = np.array([0.0])
    weights = np.array([1.0])
    matrix = np.array(
        [
            [0.0],
        ]
    )


class BackwardEuler(RungeKutta):
    """
    Backward Euler. A favorite among true connoisseurs of the heat equation.

    A-stable first order method.
    """

    nodes = np.array([0.0])
    weights = np.array([1.0])
    matrix = np.array(
        [
            [1.0],
        ]
    )


class CrankNicholson(RungeKutta):
    """
    Implicit Runge-Kutta method of second order, A-stable.
    """

    nodes = np.array([0, 1])
    weights = np.array([0.5, 0.5])
    matrix = np.zeros((2, 2))
    matrix[1, 0] = 0.5
    matrix[1, 1] = 0.5


class ExplicitMidpointMethod(RungeKutta):
    """
    Explicit Runge-Kutta method of second order.
    """

    nodes = np.array([0, 0.5])
    weights = np.array([0, 1])
    matrix = np.zeros((2, 2))
    matrix[1, 0] = 0.5


class ImplicitMidpointMethod(RungeKutta):
    """
    Implicit Runge-Kutta method of second order.
    """

    nodes = np.array([0.5])
    weights = np.array([1])
    matrix = np.zeros((1, 1))
    matrix[0, 0] = 1.0 / 2.0


class RK4(RungeKutta):
    """
    Explicit Runge-Kutta of fourth order: Everybody's darling.
    """

    nodes = np.array([0, 0.5, 0.5, 1])
    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    matrix = np.zeros((4, 4))
    matrix[1, 0] = 0.5
    matrix[2, 1] = 0.5
    matrix[3, 2] = 1.0


class Heun_Euler(RungeKutta):
    """
    Second order explicit embedded Runge-Kutta method.
    """

    nodes = np.array([0, 1])
    weights = np.array([[0.5, 0.5], [1, 0]])
    matrix = np.zeros((2, 2))
    matrix[1, 0] = 1
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 2


class Cash_Karp(RungeKutta):
    """
    Fifth order explicit embedded Runge-Kutta. See [here](https://doi.org/10.1145/79505.79507).
    """

    nodes = np.array([0, 0.2, 0.3, 0.6, 1.0, 7.0 / 8.0])
    weights = np.array(
        [
            [37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0],
            [2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0],
        ]
    )
    matrix = np.zeros((6, 6))
    matrix[1, 0] = 1.0 / 5.0
    matrix[2, :2] = [3.0 / 40.0, 9.0 / 40.0]
    matrix[3, :3] = [0.3, -0.9, 1.2]
    matrix[4, :4] = [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0]
    matrix[5, :5] = [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0]
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 5


class DIRK43(RungeKutta):
    """
    Embedded A-stable diagonally implicit RK pair of order 3 and 4.

    Taken from [here](https://doi.org/10.1007/BF01934920).
    """

    nodes = np.array([5.0 / 6.0, 10.0 / 39.0, 0, 1.0 / 6.0])
    weights = np.array(
        [[61.0 / 150.0, 2197.0 / 2100.0, 19.0 / 100.0, -9.0 / 14.0], [32.0 / 75.0, 169.0 / 300.0, 1.0 / 100.0, 0.0]]
    )
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 5.0 / 6.0
    matrix[1, :2] = [-15.0 / 26.0, 5.0 / 6.0]
    matrix[2, :3] = [215.0 / 54.0, -130.0 / 27.0, 5.0 / 6.0]
    matrix[3, :] = [4007.0 / 6075.0, -31031.0 / 24300.0, -133.0 / 2700.0, 5.0 / 6.0]
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 4


class ESDIRK53(RungeKutta):
    """
    A-stable embedded RK pair of orders 5 and 3.
    Taken from [here](https://ntrs.nasa.gov/citations/20160005923)
    """

    nodes = np.array(
        [0, 4024571134387.0 / 7237035672548.0, 14228244952610.0 / 13832614967709.0, 1.0 / 10.0, 3.0 / 50.0, 1.0]
    )
    matrix = np.zeros((6, 6))
    matrix[1, :2] = [3282482714977.0 / 11805205429139.0, 3282482714977.0 / 11805205429139.0]
    matrix[2, :3] = [
        606638434273.0 / 1934588254988,
        2719561380667.0 / 6223645057524,
        3282482714977.0 / 11805205429139.0,
    ]
    matrix[3, :4] = [
        -651839358321.0 / 6893317340882,
        -1510159624805.0 / 11312503783159,
        235043282255.0 / 4700683032009.0,
        3282482714977.0 / 11805205429139.0,
    ]
    matrix[4, :5] = [
        -5266892529762.0 / 23715740857879,
        -1007523679375.0 / 10375683364751,
        521543607658.0 / 16698046240053.0,
        514935039541.0 / 7366641897523.0,
        3282482714977.0 / 11805205429139.0,
    ]
    matrix[5, :] = [
        -6225479754948.0 / 6925873918471,
        6894665360202.0 / 11185215031699,
        -2508324082331.0 / 20512393166649,
        -7289596211309.0 / 4653106810017.0,
        39811658682819.0 / 14781729060964.0,
        3282482714977.0 / 11805205429139,
    ]

    weights = np.array(
        [
            [
                -6225479754948.0 / 6925873918471,
                6894665360202.0 / 11185215031699.0,
                -2508324082331.0 / 20512393166649,
                -7289596211309.0 / 4653106810017,
                39811658682819.0 / 14781729060964.0,
                3282482714977.0 / 11805205429139,
            ],
            [
                -2512930284403.0 / 5616797563683,
                5849584892053.0 / 8244045029872,
                -718651703996.0 / 6000050726475.0,
                -18982822128277.0 / 13735826808854.0,
                23127941173280.0 / 11608435116569.0,
                2847520232427.0 / 11515777524847.0,
            ],
        ]
    )
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 4
