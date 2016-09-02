import abc
import numpy as np

from pySDC.Collocation import CollBase
from pySDC.Level import level

from future.utils import with_metaclass


class sweeper(with_metaclass(abc.ABCMeta)):
    """
    Base abstract sweeper class

    Attributes:
        __level: current level
        coll: collocation object
    """

    def __init__(self,params):
        """
        Initialization routine for the base sweeper

        Args:
            params: parameter object

        """

        # short helper class to add params as attributes
        class pars():
            def __init__(self,params):

                defaults = dict()
                defaults['do_LU'] = False
                defaults['do_coll_update'] = False
                
                for k,v in defaults.items():
                    setattr(self,k,v)

                for k,v in params.items():
                    if not k is 'collocation_class':
                        setattr(self,k,v)

        self.params = pars(params)

        coll = params['collocation_class'](params['num_nodes'],0,1)
        assert isinstance(coll, CollBase)
        if not coll.right_is_node:
          assert self.params.do_coll_update, "For nodes where the right end point is not a node, do_coll_update has to be set to True"

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        # collocation object
        self.coll = coll


    def __set_level(self,L):
        """
        Sets a reference to the current level (done in the initialization of the level)

        Args:
            L: current level
        """
        assert isinstance(L,level)
        self.__level = L


    @property
    def level(self):
        """
        Returns the current level
        """
        return self.__level


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
        L.f[0] = P.eval_f(L.u[0],L.time)

        # copy u[0] to all collocation nodes, evaluate RHS
        for m in range(1,self.coll.num_nodes+1):
            L.u[m] = P.dtype_u(L.u[0])
            L.f[m] = P.eval_f(L.u[m],L.time+L.dt*self.coll.nodes[m-1])

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

        return None


    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if there are new values (e.g. from a sweep)
        assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            # add u0 and subtract u at current node
            res[m] += L.u[0] - L.u[m+1]
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

    @abc.abstractmethod
    def compute_end_point(self):
        """
        Abstract interface to end-node computation
        """
        return None


    @abc.abstractmethod
    def integrate(self):
        """
        Abstract interface to right-hand side integration
        """
        return None


    @abc.abstractmethod
    def update_nodes(self):
        """
        Abstract interface to node update
        """
        return None
