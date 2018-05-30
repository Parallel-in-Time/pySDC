import logging
import numpy as np
import scipy.linalg
import scipy.optimize as opt

from pySDC.core.Level import level
from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.core.Errors import ParameterError


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):

        self.do_coll_update = False
        self.spread = True

        for k, v in pars.items():
            if k is not 'collocation_class':
                setattr(self, k, v)

        self._freeze()


class sweeper(object):
    """
    Base abstract sweeper class

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

        essential_keys = ['collocation_class', 'num_nodes']
        for key in essential_keys:
            if key not in params:
                msg = 'need %s to instantiate step, only got %s' % (key, str(params.keys()))
                self.logger.error(msg)
                raise ParameterError(msg)

        self.params = _Pars(params)

        coll = params['collocation_class'](params['num_nodes'], 0, 1)

        if not coll.right_is_node and not self.params.do_coll_update:
            self.logger.warning('we need to do a collocation update here, since the right end point is not a node. '
                                'Changing this!')
            self.params.do_coll_update = True

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        # collocation object
        self.coll = coll

    def get_Qdelta_implicit(self, coll, qd_type):

        def rho(x):
            return max(abs(np.linalg.eigvals(np.eye(m) - np.diag([x[i] for i in range(m)]).dot(coll.Qmat[1:, 1:]))))

        QDmat = np.zeros(coll.Qmat.shape)
        if qd_type == 'LU':
            QT = coll.Qmat[1:, 1:].T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            QDmat[1:, 1:] = U.T
        elif qd_type == 'LU2':
            QT = coll.Qmat[1:, 1:].T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            QDmat[1:, 1:] = 2 * U.T
        elif qd_type == 'IE':
            for m in range(coll.num_nodes + 1):
                QDmat[m, 1:m + 1] = coll.delta_m[0:m]
        elif qd_type == 'IEpar':
            for m in range(coll.num_nodes + 1):
                QDmat[m, m] = np.sum(coll.delta_m[0:m])
        elif qd_type == 'Qpar':
            QDmat = np.diag(np.diag(coll.Qmat))
        elif qd_type == 'GS':
            QDmat = np.tril(coll.Qmat)
        elif qd_type == 'PIC':
            QDmat = np.zeros(coll.Qmat.shape)
        elif qd_type == 'MIN':
            m = QDmat.shape[0] - 1
            x0 = 10 * np.ones(m)
            d = opt.minimize(rho, x0, method='Nelder-Mead')
            QDmat[1:, 1:] = np.linalg.inv(np.diag(d.x))
        else:
            raise NotImplementedError('qd_type implicit not implemented')
        # check if we got not more than a lower triangular matrix
        np.testing.assert_array_equal(np.triu(QDmat, k=1), np.zeros(QDmat.shape),
                                      err_msg='Lower triangular matrix expected!')

        return QDmat

    def get_Qdelta_explicit(self, coll, qd_type):
        QDmat = np.zeros(coll.Qmat.shape)
        if qd_type == 'EE':
            for m in range(self.coll.num_nodes + 1):
                QDmat[m, 0:m] = self.coll.delta_m[0:m]
        elif qd_type == 'GS':
            QDmat = np.tril(self.coll.Qmat, k=-1)
        elif qd_type == 'PIC':
            QDmat = np.zeros(coll.Qmat.shape)
        else:
            raise NotImplementedError('qd_type explicit not implemented')

        # check if we got not more than a lower triangular matrix
        np.testing.assert_array_equal(np.triu(QDmat, k=0), np.zeros(QDmat.shape),
                                      err_msg='Strictly lower triangular matrix expected!')

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

        # copy u[0] to all collocation nodes, evaluate RHS
        for m in range(1, self.coll.num_nodes + 1):
            if self.params.spread:
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            else:
                L.u[m] = P.dtype_u(init=P.init, val=0)
                L.f[m] = P.dtype_f(init=P.init, val=0)

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            # add u0 and subtract u at current node
            res[m] += L.u[0] - L.u[m + 1]
            # add tau if associated
            if L.tau is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

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
        assert isinstance(L, level)
        self.__level = L
