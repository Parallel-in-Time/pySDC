import numpy as np
import logging

from pySDC.core.Sweeper import _Pars
from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class ButcherTableau(object):
    def __init__(self, weights, nodes, matrix):
        """
        Initialization routine for an collocation object

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
        self.num_nodes = matrix.shape[0]
        self.tleft = 0.
        self.tright = 1.

        self.nodes = np.append(nodes, [1])
        self.weights = weights
        self.Qmat = np.zeros([self.num_nodes + 1, self.num_nodes + 1])
        self.Qmat[:-1, :-1] = matrix
        self.Qmat[-1, :-1] = weights  # this is for computing the solution to the step from the previous stages

        self.left_is_node = True
        self.right_is_node = self.nodes[-1] == self.tright

        # compute distances between the nodes
        if self.num_nodes > 1:
            self.delta_m = self.nodes[1:] - self.nodes[:-1]
        else:
            self.delta_m = np.zeros(1)
        self.delta_m[0] = self.nodes[0] - self.tleft

        # check if the RK scheme is implicit
        self.implicit = any([matrix[i, i] != 0 for i in range(self.num_nodes)])


class RungeKutta(generic_implicit):
    """
    Runge-Kutta scheme that fits the interface of a sweeper.
    However, this is not a sweeper at all, since Runge-Kutta schemes are direct methods. We perform exactly one
    "iteration" of generic SDC and by choosing Q = Q_Delta = <Butcher tableau>, we can implement a Runge-Kutta to
    compare to the various flavours of SDC that we can think of.

    I want to emphasize that this sweeper does not fit the intendet purpose of the class it inherits from, but allows
    to make time measurements to compare SDC to popular schemes that are actually in use.

    This class only supports lower triangular Butcher tableaus such that the system can be solved with forward
    subsitution. In this way, we don't get the maximum order that we could for the number of stages, but computing the
    stages is much cheaper. In particular, if the Butcher tableaus is strictly lower trianglar, we get an explicit
    method, which does not require us to solve a system of equations to compute the stages.

    Attribues:
        butcher_tableau (ButcherTableau): Butcher tableau for the Runge-Kutta scheme that you want
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        # set up logger
        self.logger = logging.getLogger('sweeper')

        essential_keys = ['butcher_tableau', 'num_nodes']
        for key in essential_keys:
            if key not in params:
                msg = 'need %s to instantiate step, only got %s' % (key, str(params.keys()))
                self.logger.error(msg)
                raise ParameterError(msg)

        self.params = _Pars(params)

        self.coll = params['butcher_tableau']

        if not self.coll.right_is_node and not self.params.do_coll_update:
            self.logger.warning('we need to do a collocation update here, since the right end point is not a node. '
                                'Changing this!')
            self.params.do_coll_update = True

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        self.parallelizable = False

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

        # evaluate at the first node
        L.f[0] = P.eval_f(L.u[0], L.time)

        # compute the stages
        for m in range(0, self.coll.num_nodes):
            # initialize variable to store the solution at which to evaluate to compute the new stage
            u_temp = L.u[0]

            # add the stages to compute the solution at which to evaluate for the next stage
            for j in range(0, m + 1):
                u_temp += L.dt * self.coll.Qmat[m + 1, j] * L.f[j]

            # compute the new intermediate solution
            if self.coll.implicit:
                L.u[m + 1] = P.solve_system(u_temp, L.dt * self.coll.Qmat[m + 1, m + 1], L.u[m + 1],
                                            L.time + L.dt * self.coll.nodes[m])
            else:
                L.u[m + 1] = u_temp

            # evaluate the new stage
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None


class RK1(RungeKutta):
    def __init__(self, params):
        implicit = params.get('implicit', False)
        nodes = np.array([0.])
        weights = np.array([1.])
        if implicit:
            matrix = np.array([[1.], ])
        else:
            matrix = np.array([[0.], ])
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(RK1, self).__init__(params)


class MidpointMethod(RungeKutta):
    '''
    Runge-Kutta method of second order
    '''
    def __init__(self, params):
        implicit = params.get('implicit', False)
        nodes = np.array([0, 0.5])
        weights = np.array([0, 1])
        matrix = np.zeros((2, 2))
        if implicit:
            matrix[1, 1] = 0.5
        else:
            matrix[1, 0] = 0.5
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(MidpointMethod, self).__init__(params)


class RK4(RungeKutta):
    '''
    Explicit Runge-Kutta of fourth order: Everybodies darling.
    '''
    def __init__(self, params):
        nodes = np.array([0, 0.5, 0.5, 1])
        weights = np.array([1., 2., 2., 1.]) / 6.
        matrix = np.zeros((4, 4))
        matrix[1, 0] = 0.5
        matrix[2, 1] = 0.5
        matrix[3, 2] = 1.
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(RK4, self).__init__(params)
