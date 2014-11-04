import abc
import numpy as np

from pySDC.Collocation import CollBase
from pySDC.Level import level


class sweeper(metaclass=abc.ABCMeta):
    """
    Base abstract sweeper class

    Attributes:
        __level: current level
        coll: collocation object
    """

    def __init__(self,coll):
        """
        Initialization routine for the base sweeper

        Args:
            coll: collocation object

        """

        assert isinstance(coll, CollBase)

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
        res = []
        for m in range(self.coll.num_nodes):
            # build QF(u)
            res.append(P.dtype_u(self.integrate(self.coll.Qmat[m+1,1:])))
            # add u0 and subtract u at current node
            res[m] += L.u[0] - L.u[m+1]
            # add tau if associated
            if L.tau is not None:
                res[m] += L.tau[m]

        # use standard residual norm: ||.||_inf
        L.status.residual = np.linalg.norm(res,np.inf)

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None


    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here might be a simple copy from u[M] (if right point is a collocation node) or
        a full evaluation of the Picard formulation (if right point is not a collocation node)
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point (flag is set in collocation class)
        if self.coll.right_is_node:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            L.uend += self.integrate(self.coll.weights)
            #FIXME: do we need some sort of tau correction here as well?

        return None


    @abc.abstractmethod
    def integrate(self,weights):
        """
        Abstract interface to right-hand side integration

        Args:
            weights: integration weights
        """
        return None


    @abc.abstractmethod
    def update_nodes(self):
        """
        Abstract interface to node update
        """
        return None