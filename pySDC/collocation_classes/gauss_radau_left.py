from __future__ import division
import numpy as np
import scipy.sparse as sp

from pySDC.Collocation import CollBase


class CollGaussRadau_Left(CollBase):
    """
    Implements Gauss-Radau Quadrature by deriving from CollBase and implementing Gauss-Radau nodes
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Left, self).__init__(num_nodes, tleft, tright)
        assert num_nodes >= 2, "Number of nodes should be at least 2 for Gauss-Radau, but is %d" % num_nodes
        self.order = 2 * self.num_nodes - 1
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = False

    @property
    def _getNodes(self):
        """
        Copyright by Daniel Ruprecht (who copied this from somewhere else), 2014
        Computes Gauss-Radau integration nodes with left point included.
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        alpha = 0.0
        beta = 1.0

        diag = np.zeros(M-1)
        subdiag = np.zeros(M-2)

        diag[0] = (beta-alpha)/(2+alpha+beta)

        for jj in range(1,M-1):
            diag[jj] = (beta-alpha)*(alpha+beta)/(2*jj + 2 + alpha + beta)/(2*jj+alpha+beta)
            subdiag[jj-1] = np.sqrt( 4*jj*(jj+alpha)*(jj+beta)*(jj+alpha+beta) ) \
                         / np.sqrt( (2*jj-1+alpha+beta)*(2*jj+alpha+beta)**2*(2*jj+1+alpha+beta))

        subdiag1 = np.zeros(M-1)
        subdiag2 = np.zeros(M-1)
        subdiag1[0:-1] = subdiag
        subdiag2[1:] = subdiag

        Mat = sp.spdiags(data=[subdiag1,diag,subdiag2],diags=[-1,0,1],m=M-1,n=M-1).todense()

        x = np.sort(np.linalg.eigvals(Mat))

        nodes = np.concatenate(([-1.0],x))

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2
        print('WARNING: GaussRadau_Left not fully tested, use at own risk!')

        return nodes
