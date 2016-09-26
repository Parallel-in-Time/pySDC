from __future__ import division
import numpy as np

from pySDC.Collocation import CollBase

class EquidistantInner(CollBase):
    """
    Implements equidistant nodes with both boundary points excluded by deriving from CollBase
    """
    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantInner, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 1 for EquidistantInner, but is %d" % num_nodes
        self.order = self.num_nodes
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = False

    @property
    def _getNodes(self):
        """
        Computes integration nodes with both boundaries excluded
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright
        nodes = np.linspace(a, b, M+1, endpoint=False)
        nodes = nodes[1:]
        return nodes