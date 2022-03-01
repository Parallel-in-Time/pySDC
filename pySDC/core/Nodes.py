import numpy as np
from scipy.linalg import eigh_tridiagonal

NODE_TYPES = ['EQUID', 'LEGENDRE',
              'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']

QUAD_TYPES = ['GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO']


class NodesError(Exception):
    pass


class NodesGenerator(object):

    def __init__(self, node_type='LEGENDRE', quad_type='LOBATTO'):

        # Check argument validity
        for arg, vals in zip(['node_type', 'quad_type'],
                             [NODE_TYPES, QUAD_TYPES]):
            val = eval(arg)
            if val not in vals:
                raise NodesError(
                    f"{arg}='{val}' not implemented, must be in {vals}")

        # Store attributes
        self.node_type = node_type
        self.quad_type = quad_type

    def getNodes(self, num_nodes):
        # Check number of nodes
        if self.quad_type in ['LOBATTO', 'RADAU-LEFT'] and num_nodes < 2:
            raise NodesError(
                f"num_nodes must be larger than 2 for {self.quad_type}, "
                f"but for {num_nodes}")
        elif num_nodes < 1:
            raise NodesError("you surely want at least one node ;)")

        # Equidistant nodes
        if self.node_type == 'EQUID':
            if self.quad_type == 'GAUSS':
                return np.linspace(-1, 1, num=num_nodes + 2)[1:-1]
            elif self.quad_type == 'LOBATTO':
                return np.linspace(-1, 1, num=num_nodes)
            elif self.quad_type == 'RADAU-RIGHT':
                return np.linspace(-1, 1, num=num_nodes + 1)[1:]
            elif self.quad_type == 'RADAU-LEFT':
                return np.linspace(-1, 1, num=num_nodes + 1)[:-1]

        # Quadrature nodes linked to orthogonal polynomials
        alpha, beta = self.getTridiagCoefficients(num_nodes)
        nodes = eigh_tridiagonal(alpha, np.sqrt(beta[1:]))[0]
        nodes.sort()

        return nodes

    def getOrthogPolyCoefficients(self, num_coeff):
        if self.node_type == 'LEGENDRE':
            k = np.arange(num_coeff, dtype=float)
            alpha = 0 * k
            beta = k**2 / (4 * k**2 - 1)
            beta[0] = 2
        elif self.node_type == 'CHEBY-1':
            alpha = np.zeros(num_coeff)
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
            if num_coeff > 1:
                beta[1] = 0.5
        elif self.node_type == 'CHEBY-2':
            alpha = np.zeros(num_coeff)
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi / 2
        elif self.node_type == 'CHEBY-3':
            alpha = np.zeros(num_coeff)
            alpha[0] = 0.5
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
        elif self.node_type == 'CHEBY-4':
            alpha = np.zeros(num_coeff)
            alpha[0] = -0.5
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
        return alpha, beta

    def evalOrthogPoly(self, t, alpha, beta):
        """
        Evaluate the two higher order orthogonal polynomials corresponding
        to the given (alpha,beta) coefficients.

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        beta : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        """
        t = np.asarray(t, dtype=float)
        pi = np.array([np.zeros_like(t) for i in range(3)])
        pi[1:] += 1
        for alpha_j, beta_j in zip(alpha, beta):
            pi[2] *= (t - alpha_j)
            pi[0] *= beta_j
            pi[2] -= pi[0]
            pi[0] = pi[1]
            pi[1] = pi[2]
        return pi[0], pi[1]

    def getTridiagCoefficients(self, num_nodes):
        # Coefficients for Gauss quadrature type
        alpha, beta = self.getOrthogPolyCoefficients(num_nodes)

        # If not Gauss quadrature type, modify the alpha/beta coefficients
        if self.quad_type.startswith('RADAU'):
            b = -1. if self.quad_type.endswith('LEFT') else 1.
            b1, b2 = self.evalOrthogPoly(b, alpha[:-1], beta[:-1])[:2]
            alpha[-1] = b - beta[-1] * b1 / b2
        elif self.quad_type == 'LOBATTO':
            a, b = -1., 1.
            a2, a1 = self.evalOrthogPoly(a, alpha[:-1], beta[:-1])[:2]
            b2, b1 = self.evalOrthogPoly(b, alpha[:-1], beta[:-1])[:2]
            alpha[-1], beta[-1] = np.linalg.solve(
                [[a1, a2],
                 [b1, b2]],
                [a * a1, b * b1])
        return alpha, beta
