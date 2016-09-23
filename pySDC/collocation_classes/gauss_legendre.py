from __future__ import division
import numpy as np

from pySDC.Collocation import CollBase

class CollGaussLegendre(CollBase):
    """
    Implements Gauss-Legendre Quadrature by deriving from CollBase and implementing Gauss-Legendre nodes
    -> actually already part of CollBase, this is just for consistency
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLegendre, self).__init__(num_nodes, tleft, tright)
        assert num_nodes >= 1, "Number of nodes should be at least 1 for Gauss-Legendre, but is %d" % num_nodes
        self.order = 2 * self.num_nodes
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
        Computes nodes for the Gauss-Legendre quadrature of order :math:`n>1` on :math:`[-1,+1]`.

        (ported from MATLAB code, reference see below, original commend from MATLAB code:)

        .. epigraph::

            Unlike many publicly available functions, this function is valid for :math:`n>=46`.
            This is due to the fact that it does not rely on MATLAB's build-in 'root' routines to determine the roots
            of the Legendre polynomial, but finds the roots by looking for the eigenvalues of an alternative version of
            the companion matrix of the n'th degree Legendre polynomial.
            The companion matrix is constructed as a symmetrical matrix, guaranteeing that all the eigenvalues (roots)
            will be real.
            On the contrary, MATLAB's 'roots' function uses a general form for the companion matrix, which becomes
            unstable at higher orders :math:`n`, leading to complex roots.

            -- original MATLAB function by: Geert Van Damme <geert@vandamme-iliano.be> (February 21, 2010)
        Python version by Dieter Moser, 2014
        :return: Gauss-Legendre nodes
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        # Building the companion matrix comp_mat with det(nodes*I-comp_mat)=P_n(nodes), where P_n is the
        # Legendre polynomial under consideration. comp_mat will be constructed in such a way that it is symmetric.
        linspace = np.linspace(1, M - 1, M - 1)
        v = [linspace[i] / np.sqrt(4.0 * linspace[i] ** 2 - 1.0) for i in range(len(linspace))]
        comp_mat = np.diag(v, 1) + np.diag(v, -1)

        # Determining the abscissas (nodes) - since det(nodesI-comp_mat)=P_n(nodes), the abscissas are the roots
        # of the characteristic polynomial, i.e. the eigenvalues of comp_mat
        [eig_vals, _] = np.linalg.eig(comp_mat)
        indizes = np.argsort(eig_vals)
        nodes = eig_vals[indizes]

        # take real part and shift from [-1,1] to [a,b]
        nodes = nodes.real
        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        return nodes