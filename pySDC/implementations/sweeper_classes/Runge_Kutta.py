import numpy as np
import logging
from qmat.qcoeff.butcher import RK_SCHEMES

from pySDC.core.sweeper import Sweeper, _Pars
from pySDC.core.errors import ParameterError
from pySDC.core.level import Level


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


class RungeKutta(Sweeper):
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
        if type(f).__name__ in ['mesh', 'cupy_mesh']:
            return f
        elif type(f).__name__ in ['imex_mesh', 'imex_cupy_mesh']:
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
            rhs = prob.dtype_u(lvl.u[0])
            for j in range(1, m + 1):
                rhs += lvl.dt * self.QI[m + 1, j] * self.get_full_f(lvl.f[j])

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            if self.QI[m + 1, m + 1] != 0:
                lvl.u[m + 1][:] = prob.solve_system(
                    rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
                )
            else:
                lvl.u[m + 1][:] = rhs[:]

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            if m < M - self.coll.num_solution_stages or self.params.eval_rhs_at_right_boundary:
                lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])

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
        assert isinstance(lvl, Level), f"You tried to set the sweeper's level with an instance of {type(lvl)}!"
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


class RungeKuttaIMEX(RungeKutta):
    """
    Implicit-explicit split Runge Kutta base class. Only supports methods that share the nodes and weights.
    """

    matrix_explicit = None
    ButcherTableauClass_explicit = ButcherTableau

    def __init__(self, params):
        """
        Initialization routine

        Args:
            params: parameters for the sweeper
        """
        super().__init__(params)
        self.coll_explicit = self.get_Butcher_tableau_explicit()
        self.QE = self.coll_explicit.Qmat

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        for m in range(1, self.coll.num_nodes + 1):
            lvl.u[m] = prob.dtype_u(init=prob.init, val=0.0)
            lvl.f[m] = prob.dtype_f(init=prob.init, val=0.0)

        # indicate that this level is now ready for sweeps
        lvl.status.unlocked = True
        lvl.status.updated = True

    @classmethod
    def get_Butcher_tableau_explicit(cls):
        return cls.ButcherTableauClass_explicit(cls.weights, cls.nodes, cls.matrix_explicit)

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
                me[-1] += lvl.dt * (
                    self.coll.Qmat[m, j] * lvl.f[j].impl + self.coll_explicit.Qmat[m, j] * lvl.f[j].expl
                )

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
                rhs += lvl.dt * (self.QI[m + 1, j] * lvl.f[j].impl + self.QE[m + 1, j] * lvl.f[j].expl)

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            if self.QI[m + 1, m + 1] != 0:
                lvl.u[m + 1][:] = prob.solve_system(
                    rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
                )
            else:
                lvl.u[m + 1][:] = rhs[:]

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            if m < M - self.coll.num_solution_stages or self.params.eval_rhs_at_right_boundary:
                lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None


class ForwardEuler(RungeKutta):
    """
    Forward Euler. Still a classic.

    Not very stable first order method.
    """

    generator = RK_SCHEMES["FE"]()
    nodes, weights, matrix = generator.genCoeffs()


class BackwardEuler(RungeKutta):
    """
    Backward Euler. A favorite among true connoisseurs of the heat equation.

    A-stable first order method.
    """

    generator = RK_SCHEMES["BE"]()
    nodes, weights, matrix = generator.genCoeffs()


class CrankNicolson(RungeKutta):
    """
    Implicit Runge-Kutta method of second order, A-stable.
    """

    generator = RK_SCHEMES["CN"]()
    nodes, weights, matrix = generator.genCoeffs()


class ExplicitMidpointMethod(RungeKutta):
    """
    Explicit Runge-Kutta method of second order.
    """

    generator = RK_SCHEMES["RK2"]()
    nodes, weights, matrix = generator.genCoeffs()


class ImplicitMidpointMethod(RungeKutta):
    """
    Implicit Runge-Kutta method of second order.
    """

    generator = RK_SCHEMES["IMP"]()
    nodes, weights, matrix = generator.genCoeffs()


class RK4(RungeKutta):
    """
    Explicit Runge-Kutta of fourth order: Everybody's darling.
    """

    generator = RK_SCHEMES["RK4"]()
    nodes, weights, matrix = generator.genCoeffs()


class Heun_Euler(RungeKutta):
    """
    Second order explicit embedded Runge-Kutta method.
    """

    generator = RK_SCHEMES["HEUN"]()
    nodes, weights, matrix = generator.genCoeffs()

    @classmethod
    def get_update_order(cls):
        return 2


class Cash_Karp(RungeKutta):
    """
    Fifth order explicit embedded Runge-Kutta. See [here](https://doi.org/10.1145/79505.79507).
    """

    generator = RK_SCHEMES["CashKarp"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 5


class DIRK43(RungeKutta):
    """
    Embedded A-stable diagonally implicit RK pair of order 3 and 4.

    Taken from [here](https://doi.org/10.1007/BF01934920).
    """

    generator = RK_SCHEMES["EDIRK43"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 4


class DIRK43_2(RungeKutta):
    """
    L-stable Diagonally Implicit RK method with four stages of order 3.
    Taken from [here](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
    """

    generator = RK_SCHEMES["DIRK43"]()
    nodes, weights, matrix = generator.genCoeffs()


class EDIRK4(RungeKutta):
    """
    Stiffly accurate, fourth-order EDIRK with four stages. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (216).
    """

    generator = RK_SCHEMES["EDIRK4"]()
    nodes, weights, matrix = generator.genCoeffs()


class ESDIRK53(RungeKutta):
    """
    A-stable embedded RK pair of orders 5 and 3, ESDIRK5(3)6L[2]SA.
    Taken from [here](https://ntrs.nasa.gov/citations/20160005923)
    """

    generator = RK_SCHEMES["ESDIRK53"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 4


class ESDIRK43(RungeKutta):
    """
    A-stable embedded RK pair of orders 4 and 3, ESDIRK4(3)6L[2]SA.
    Taken from [here](https://ntrs.nasa.gov/citations/20160005923)
    """

    generator = RK_SCHEMES["ESDIRK43"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 4


class ARK548L2SAERK(RungeKutta):
    """
    Explicit part of the ARK54 scheme.
    """

    generator = RK_SCHEMES["ARK548L2SAERK"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 5


class ARK548L2SAESDIRK(ARK548L2SAERK):
    """
    Implicit part of the ARK54 scheme. Be careful with the embedded scheme. It seems that both schemes are order 5 as opposed to 5 and 4 as claimed. This may cause issues when doing adaptive time-stepping.
    """

    generator_IMP = RK_SCHEMES["ARK548L2SAESDIRK"]()
    matrix = generator_IMP.Q


class ARK54(RungeKuttaIMEX):
    """
    Pair of pairs of ARK5(4)8L[2]SA-ERK and ARK5(4)8L[2]SA-ESDIRK from [here](https://doi.org/10.1016/S0168-9274(02)00138-1).
    """

    ButcherTableauClass = ButcherTableauEmbedded
    ButcherTableauClass_explicit = ButcherTableauEmbedded

    nodes = ARK548L2SAERK.nodes
    weights = ARK548L2SAERK.weights

    matrix = ARK548L2SAESDIRK.matrix
    matrix_explicit = ARK548L2SAERK.matrix

    @classmethod
    def get_update_order(cls):
        return 5


class ARK548L2SAESDIRK2(RungeKutta):
    """
    Stiffly accurate singly diagonally L-stable implicit embedded Runge-Kutta pair of orders 5 and 4 with explicit first stage from [here](https://doi.org/10.1016/j.apnum.2018.10.007).
    This method is part of the IMEX method ARK548L2SA.
    """

    generator = RK_SCHEMES["ARK548L2SAESDIRK2"]()
    nodes, weights, matrix = generator.genCoeffs(embedded=True)
    ButcherTableauClass = ButcherTableauEmbedded

    @classmethod
    def get_update_order(cls):
        return 5


class ARK548L2SAERK2(ARK548L2SAESDIRK2):
    """
    Explicit embedded pair of Runge-Kutta methods of orders 5 and 4 from [here](https://doi.org/10.1016/j.apnum.2018.10.007).
    This method is part of the IMEX method ARK548L2SA.
    """

    generator_EXP = RK_SCHEMES["ARK548L2SAERK2"]()
    matrix = generator_EXP.Q


class ARK548L2SA(RungeKuttaIMEX):
    """
    IMEX Runge-Kutta method of order 5 based on the explicit method ARK548L2SAERK2 and the implicit method
    ARK548L2SAESDIRK2 from [here](https://doi.org/10.1016/j.apnum.2018.10.007).

    According to Kennedy and Carpenter (see reference), the two IMEX RK methods of order 5 are the only ones available
    as of now. And we are not aware of higher order ones. This one is newer then the other one and apparently better.
    """

    ButcherTableauClass = ButcherTableauEmbedded
    ButcherTableauClass_explicit = ButcherTableauEmbedded

    nodes = ARK548L2SAERK2.nodes
    weights = ARK548L2SAERK2.weights

    matrix = ARK548L2SAESDIRK2.matrix
    matrix_explicit = ARK548L2SAERK2.matrix

    @classmethod
    def get_update_order(cls):
        return 5
