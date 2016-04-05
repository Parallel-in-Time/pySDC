from __future__ import division
import numpy as np
import scipy.sparse as sp
import numpy.polynomial.legendre as leg
from scipy.linalg import lu
import scipy.interpolate as intpl
from pySDC.Collocation import CollBase


class CollGaussLegendre(CollBase):
    """
    Implements Gauss-Legendre Quadrature by deriving from CollBase and implementing Gauss-Legendre nodes
    -> actually already part of CollBase, this is just for consistency
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLegendre, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 1 for Gauss-Legendre, but is %d" % num_nodes
        self.order = 2 * self.num_nodes
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.QDmat = self._gen_QDmatrix
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
        v = linspace / np.sqrt(4.0 * linspace ** 2 - 1.0)
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


class CollGaussLobatto(CollBase):
    """
    Implements Gauss-Lobatto Quadrature by deriving from CollBase and implementing Gauss-Lobatto nodes
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLobatto, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 2 for Gauss-Lobatto, but is %d" % num_nodes
        self.order = 2 * self.num_nodes - 2
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = True
        self.QDmat = self._gen_QDmatrix

    @property
    def _getNodes(self):
        """
        Copyright by Dieter Moser, 2014
        Computes Gauss-Lobatto integration nodes.

        Calculates the Gauss-Lobatto integration nodes via a root calculation of derivatives of the legendre
        polynomials. Note that the precision of float 64 is not guarantied.
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        roots = leg.legroots(leg.legder(np.array([0] * (M - 1) + [1], dtype=np.float64)))
        nodes = np.array(np.append([-1.0], np.append(roots, [1.0])), dtype=np.float64)

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        return nodes


class CollGaussRadau_Right(CollBase):
    """
    Implements Gauss-Radau Quadrature by deriving from CollBase and implementing Gauss-Radau nodes
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Right, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 2 for Gauss-Radau, but is %d" % num_nodes
        self.order = 2 * self.num_nodes - 1
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = True
        self.QDmat = self._gen_QDmatrix
    @property
    def _getNodes(self):
        """
        Copyright by Daniel Ruprecht (who copied this from somewhere else), 2014
        Computes Gauss-Radau integration nodes with right point included.
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        alpha = 1.0
        beta = 0.0

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

        nodes = np.concatenate((x,[1.0]))

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2


        return nodes


class CollGaussRadau_Left(CollBase):
    """
    Implements Gauss-Radau Quadrature by deriving from CollBase and implementing Gauss-Radau nodes
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Left, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 2 for Gauss-Radau, but is %d" % num_nodes
        self.order = 2 * self.num_nodes - 1
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = False
        self.QDmat = self._gen_QDmatrix

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
        print('WARNING: GaussRadau_Left untested, use at own risk!')

        return nodes

class CollGaussRadau_Right_LU_Trick(CollGaussRadau_Right):
    """
    Implements Gauss-Radau Quadrature by deriving from CollBase and implementing Gauss-Radau nodes and as
    preconditioner we implement the LU_Trick
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Right_LU_Trick, self).__init__(num_nodes, tleft, tright)

        Q = self.Qmat

        p, l, u = lu(Q[1:, 1:].transpose())
        #print np.diag(l)
        self.QDmat = u.transpose()


class CollSplineRight(CollBase):
    """
    If a spectral quadrature method is used a order higher than 15 is not applicable,
    because the underlying interpolation is numerically losses the stability. This collocation class
    uses spline functions to achieve arbitrary big Q matrices with a band structure.
    """

    def __init__(self, num_nodes, tleft, tright, order=3):
        super(CollSplineRight, self).__init__(num_nodes, tleft, tright)
        self.Q = np.zeros((num_nodes, num_nodes))
        self.nodes = self._getNodes

        # get the defining tck's for each spline basis function
        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        self.tcks = []
        for i in range(self.num_nodes):
            tck = intpl.splrep(self.nodes, np.roll(circ_one, i), xb=tleft, xe=tright, k=order, s=0.0)
            self.tcks.append(tck)

        self.order = order
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = True
        self.QDmat = self._gen_QDmatrix

    @property
    def _getNodes(self):
        return np.linspace(self.tleft + 1.0 / self.num_nodes, self.tright, self.num_nodes, endpoint=True)

    def _getWeights(self, a, b):
        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = intpl.splint(a, b, self.tcks[i])

        return weights
