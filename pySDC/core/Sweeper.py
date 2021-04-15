import logging

import numpy as np
import scipy.linalg
import scipy.optimize as opt

from pySDC.core.Errors import ParameterError
from pySDC.core.Level import level
from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.implementations.collocation_classes.equidistant_right import EquidistantNoLeft
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):

        self.do_coll_update = False
        self.initial_guess = 'spread'

        for k, v in pars.items():
            if k != 'collocation_class':
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

        self.parallelizable = False

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
        elif qd_type == 'TRAP':
            for m in range(coll.num_nodes + 1):
                QDmat[m, 1:m + 1] = coll.delta_m[0:m]
            for m in range(coll.num_nodes + 1):
                QDmat[m, 0:m] += coll.delta_m[0:m]
            QDmat /= 2.0
        elif qd_type == 'IE':
            for m in range(coll.num_nodes + 1):
                QDmat[m, 1:m + 1] = coll.delta_m[0:m]
        elif qd_type == 'IEpar':
            for m in range(coll.num_nodes + 1):
                QDmat[m, m] = np.sum(coll.delta_m[0:m])
            self.parallelizable = True
        elif qd_type == 'Qpar':
            QDmat = np.diag(np.diag(coll.Qmat))
            self.parallelizable = True
        elif qd_type == 'GS':
            QDmat = np.tril(coll.Qmat)
        elif qd_type == 'PIC':
            QDmat = np.zeros(coll.Qmat.shape)
            self.parallelizable = True
        elif qd_type == 'MIN':
            m = QDmat.shape[0] - 1
            x0 = 10 * np.ones(m)
            d = opt.minimize(rho, x0, method='Nelder-Mead')
            QDmat[1:, 1:] = np.linalg.inv(np.diag(d.x))
            self.parallelizable = True
        elif qd_type == 'MIN3':
            m = QDmat.shape[0] - 1
            x = None
            # These values have been obtained using Indie Solver, a commercial solver for black-box optimization which
            # aggregates several state-of-the-art optimization methods (free academic subscription plan)
            # objective function: sum over 17^2 values of lamdt, real and imaginary (WORKS SURPRISINGLY WELL!)
            if type(coll) == CollGaussLobatto:
                if m == 9:
                    # rho = 0.154786693955
                    x = [0.0, 0.14748983547536937, 0.1243753767395874, 0.08797965969063823, 0.03249792877433364,
                         0.06171633442251176, 0.08995295998705832, 0.1080641868728824, 0.11621787232558443]
                elif m == 7:
                    # rho = 0.0979351256833
                    x = [0.0, 0.18827968699454273, 0.1307213945012976, 0.04545003319140543, 0.08690617895312261,
                         0.12326429119922168, 0.13815746843252427]
                elif m == 5:
                    # rho = 0.0513543155235
                    x = [0.0, 0.2994085231050721, 0.07923154575177252, 0.14338847088077, 0.17675509273708057]
                elif m == 4:
                    # rho = 0.0381589713397
                    x = [0.0, 0.2865524188780046, 0.11264992497015984, 0.2583063168320655]
                elif m == 3:
                    # rho = 0.013592619664
                    x = [0.0, 0.2113181799416633, 0.3943250920445912]
                elif m == 2:
                    # rho = 0
                    x = [0.0, 0.5]
                else:
                    NotImplementedError('This combination of preconditioner, node type and node number is not '
                                        'implemented')
            elif type(coll) == CollGaussRadau_Right:
                if m == 9:
                    # rho = 0.151784861385
                    x = [0.14208076083211416, 0.1288153963623986, 0.10608601069476883, 0.07509520272252024,
                         0.027986167728305308, 0.05351160749903067, 0.07911315989747868, 0.09514844658836666,
                         0.10204992319487571]
                elif m == 7:
                    # rho = 0.116400161888
                    x = [0.15223871397682717, 0.12625448001038536, 0.08210714764924298, 0.03994434742760019,
                         0.1052662547386142, 0.14075805578834127, 0.15636085758812895]
                elif m == 5:
                    # rho = 0.0783352996958 (iteration 5355)
                    x = [0.2818591930905709, 0.2011358490453793, 0.06274536689514164, 0.11790265267514095,
                         0.1571629578515223]
                elif m == 4:
                    # rho = 0.057498908343
                    x = [0.3198786751412953, 0.08887606314792469, 0.1812366328324738, 0.23273925017954]
                elif m == 3:
                    # rho = 0.038744192979 (iteration 11188)
                    x = [0.3203856825077055, 0.1399680686269595, 0.3716708461097372]
                elif m == 2:
                    # rho = 0.0208560702294 (iteration 6690)
                    x = [0.2584092406077449, 0.6449261740461826]
                else:
                    NotImplementedError('This combination of preconditioner, node type and node number is not '
                                        'implemented')
            elif type(coll) == EquidistantNoLeft:
                if m == 9:
                    # rho = 0.251820022583 (iteration 32402)
                    x = [0.04067333763109274, 0.06893408176924318, 0.0944460427779633, 0.11847528720123894,
                         0.14153236351607695, 0.1638856774260845, 0.18569759470199648, 0.20707543960267513,
                         0.2280946565716198]
                elif m == 7:
                    # rho = 0.184582997611 (iteration 44871)
                    x = [0.0582690792096515, 0.09937620459067688, 0.13668728443669567, 0.1719458323664216,
                         0.20585615258818232, 0.2387890485242656, 0.27096908017041393]
                elif m == 5:
                    # rho = 0.118441339197 (iteration 34581)
                    x = [0.0937126798932547, 0.1619131388001843, 0.22442341539247537, 0.28385142992912565,
                         0.3412523013467262]
                elif m == 4:
                    # rho = 0.0844043254542 (iteration 33099)
                    x = [0.13194852204686872, 0.2296718892453916, 0.3197255970017318, 0.405619746972393]
                elif m == 3:
                    # rho = 0.0504635143866 (iteration 9783)
                    x = [0.2046955744931575, 0.3595744268324041, 0.5032243650307717]
                elif m == 2:
                    # rho = 0.0214806480623 (iteration 6109)
                    x = [0.3749891032632652, 0.6666472946796036]
                else:
                    NotImplementedError('This combination of preconditioner, node type and node number is not '
                                        'implemented')
            else:
                NotImplementedError('This combination of preconditioner, node type and node number is not '
                                    'implemented')
            QDmat[1:, 1:] = np.diag(x)
            self.parallelizable = True
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

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # start with zero everywhere
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level
        P = L.prob

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
            raise ParameterError(f'residual_type = {L.params.residual_type} not implemented, choose '
                                 f'full_abs, last_abs, full_rel or last_rel instead')

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
