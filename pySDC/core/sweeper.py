import logging
import numpy as np
from qmat import QDELTA_GENERATORS

from pySDC.core.errors import ParameterError
from pySDC.core.level import Level
from pySDC.core.collocation import CollBase
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):
        self.do_coll_update = False
        self.initial_guess = 'spread'  # default value (see also below)
        self.skip_residual_computation = ()  # gain performance at the cost of correct residual output

        for k, v in pars.items():
            if k != 'collocation_class':
                setattr(self, k, v)

        self._freeze()


class Sweeper(object):
    """
    Base abstract sweeper class, provides two base methods to generate QDelta matrices:

    - get_Qdelta_implicit(qd_type):
        Returns a (pySDC-type) QDelta matrix of **implicit type**,
        *i.e* lower triangular with zeros on the first collumn.
    - get_Qdelta_explicit(qd_type):
        Returns a (pySDC-type) QDelta matrix of **explicit type**,
        *i.e* strictly lower triangular with first node distance to zero on the first collumn.


    All possible QDelta matrix coefficients are generated with
    `qmat <https://qmat.readthedocs.io/en/latest/autoapi/qmat/qdelta/index.html>`_,
    check it out to see all available coefficient types.

    Attributes:
        logger: custom logger for sweeper-related logging
        params (__Pars): parameter object containing the custom parameters passed by the user
        coll (pySDC.Collocation.CollBase): collocation object
    """

    def __init__(self, params):
        """
        Initialization routine for the base sweeper

        Args:
            params (dict): parameter object

        """

        # set up logger
        self.logger = logging.getLogger('sweeper')

        essential_keys = ['num_nodes']
        for key in essential_keys:
            if key not in params:
                msg = 'need %s to instantiate step, only got %s' % (key, str(params.keys()))
                self.logger.error(msg)
                raise ParameterError(msg)

        if 'collocation_class' not in params:
            params['collocation_class'] = CollBase

        # prepare random generator for initial guess
        if params.get('initial_guess', 'spread') == 'random':  # default value (see also above)
            params['random_seed'] = params.get('random_seed', 1984)
            self.rng = np.random.RandomState(params['random_seed'])

        self.params = _Pars(params)

        self.coll: CollBase = params['collocation_class'](**params)

        if not self.coll.right_is_node and not self.params.do_coll_update:
            self.logger.warning(
                'we need to do a collocation update here, since the right end point is not a node. Changing this!'
            )
            self.params.do_coll_update = True

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        self.parallelizable = False

    def setupGenerator(self, qd_type):
        coll = self.coll
        try:
            assert QDELTA_GENERATORS[qd_type] == type(self.generator)
            assert self.generator.QDelta.shape[0] == coll.Qmat.shape[0] - 1
        except (AssertionError, AttributeError):
            self.generator = QDELTA_GENERATORS[qd_type](
                # for algebraic types (LU, ...)
                Q=coll.generator.Q,
                # for MIN in tables, MIN-SR-S ...
                nNodes=coll.num_nodes,
                nodeType=coll.node_type,
                quadType=coll.quad_type,
                # for time-stepping types, MIN-SR-NS
                nodes=coll.nodes,
                tLeft=coll.tleft,
            )
        except Exception as e:
            raise ValueError(f"could not generate {qd_type=!r} with qmat, got error : {e}")

    def get_Qdelta_implicit(self, qd_type, k=None):
        QDmat = np.zeros_like(self.coll.Qmat)
        self.setupGenerator(qd_type)
        QDmat[1:, 1:] = self.generator.genCoeffs(k=k)

        err_msg = 'Lower triangular matrix expected!'
        np.testing.assert_array_equal(np.triu(QDmat, k=1), np.zeros(QDmat.shape), err_msg=err_msg)
        if np.allclose(np.diag(np.diag(QDmat)), QDmat):
            self.parallelizable = True
        return QDmat

    def get_Qdelta_explicit(self, qd_type, k=None):
        coll = self.coll
        QDmat = np.zeros(coll.Qmat.shape, dtype=float)
        self.setupGenerator(qd_type)
        QDmat[1:, 1:], QDmat[1:, 0] = self.generator.genCoeffs(k=k, dTau=True)

        err_msg = 'Strictly lower triangular matrix expected!'
        np.testing.assert_array_equal(np.triu(QDmat, k=0), np.zeros(QDmat.shape), err_msg=err_msg)
        if np.allclose(np.diag(np.diag(QDmat)), QDmat):
            self.parallelizable = True  # for PIC ;)
        return QDmat

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # copy u[0] and RHS evaluation to all collocation nodes
            elif self.params.initial_guess == 'copy':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(L.f[0])
            # start with zero everywhere
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=self.rng.rand(1)[0])
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def compute_residual(self, stage=''):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            res[m] += L.u[0] - L.u[m + 1]
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(res_norm) / abs(L.u[0])
        elif L.params.residual_type == 'last_rel':
            L.status.residual = res_norm[-1] / abs(L.u[0])
        else:
            raise ParameterError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                f'full_abs, last_abs, full_rel or last_rel instead'
            )

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None

    def compute_end_point(self):
        """
        Abstract interface to end-node computation
        """
        raise NotImplementedError('ERROR: sweeper has to implement compute_end_point(self)')

    def integrate(self):
        """
        Abstract interface to right-hand side integration
        """
        raise NotImplementedError('ERROR: sweeper has to implement integrate(self)')

    def update_nodes(self):
        """
        Abstract interface to node update
        """
        raise NotImplementedError('ERROR: sweeper has to implement update_nodes(self)')

    @property
    def level(self):
        """
        Returns the current level

        Returns:
            pySDC.Level.level: the current level
        """
        return self.__level

    @level.setter
    def level(self, L):
        """
        Sets a reference to the current level (done in the initialization of the level)

        Args:
            L (pySDC.Level.level): current level
        """
        assert isinstance(L, Level)
        self.__level = L

    @property
    def rank(self):
        return 0
