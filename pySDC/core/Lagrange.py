import numpy as np
from scipy.special import roots_legendre


def computeFejerRule(n):
    """
    Compute a Fejer rule of the first kind, using DFT (Waldvogel 2006)
    Inspired from quadpy (https://github.com/nschloe/quadpy @Nico_Schlömer)

    Parameters
    ----------
    n : int
        Number of points for the quadrature rule.

    Returns
    -------
    nodes : np.1darray(n)
        The nodes of the quadrature rule
    weights : np.1darray(n)
        The weights of the quadrature rule.
    """
    # Initialize output variables
    n = int(n)
    nodes = np.empty(n, dtype=float)
    weights = np.empty(n, dtype=float)

    # Compute nodes
    theta = np.arange(1, n + 1, dtype=float)[-1::-1]
    theta *= 2
    theta -= 1
    theta *= np.pi / (2 * n)
    np.cos(theta, out=nodes)

    # Compute weights
    # -- Initial variables
    N = np.arange(1, n, 2)
    lN = len(N)
    m = n - lN
    K = np.arange(m)
    # -- Build v0
    v0 = np.concatenate([2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K**2), np.zeros(lN + 1)])
    # -- Build v1 from v0
    v1 = np.empty(len(v0) - 1, dtype=complex)
    np.conjugate(v0[:0:-1], out=v1)
    v1 += v0[:-1]
    # -- Compute inverse fourier transform
    w = np.fft.ifft(v1)
    if max(w.imag) > 1.0e-15:
        raise ValueError(f'Max imaginary value to important for ifft: {max(w.imag)}')
    # -- Store weights
    weights[:] = w.real

    return nodes, weights


class LagrangeApproximation(object):
    r"""
    Class approximating any function on a given set of points using barycentric
    Lagrange interpolation.

    Let note :math:`(t_j)_{0\leq j<n}` the set of points, then any scalar
    function :math:`f` can be approximated by the barycentric formula :

    .. math::
        p(x) =
        \frac{\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}f_j}
        {\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}},

    where :math:`f_j=f(t_j)` and

    .. math::
        w_j = \frac{1}{\prod_{k\neq j}(x_j-x_k)}

    are the barycentric weights.
    The theory and implementation is inspired from `this paper <http://dx.doi.org/10.1137/S0036144502417715>`_.

    Attributes
    ----------
    points : np.1darray
        The interpolating points
    weights : np.1darray
        The associated barycentric weights
    """

    def __init__(self, points):
        """

        Parameters
        ----------
        points : list, tuple or np.1darray
            The given interpolation points, no specific scaling, but must be
            ordered in increasing order.
        """
        points = np.asarray(points).ravel()

        diffs = points[:, None] - points[None, :]
        diffs[np.diag_indices_from(diffs)] = 1

        def analytic(diffs):
            # Fast implementation (unstable for large number of points)
            invProd = np.prod(diffs, axis=1)
            invProd **= -1
            return invProd

        with np.errstate(divide='raise', over='ignore'):
            try:
                weights = analytic(diffs)
            except FloatingPointError:
                raise ValueError('Lagrange formula unstable for that much nodes')

        # Store attributes
        self.points = points
        self.weights = weights

    @property
    def n(self):
        return self.points.size

    def getInterpolationMatrix(self, times):
        r"""
        Compute the interpolation matrix for a given set of discrete "time"
        points.

        For instance, if we note :math:`\vec{f}` the vector containing the
        :math:`f_j=f(t_j)` values, and :math:`(\tau_m)_{0\leq m<M}`
        the "time" points where to interpolate.
        Then :math:`I[\vec{f}]`, the vector containing the interpolated
        :math:`f(\tau_m)` values, can be obtained using :

        .. math::
            I[\vec{f}] = P_{Inter} \vec{f},

        where :math:`P_{Inter}` is the interpolation matrix returned by this
        method.

        Parameters
        ----------
        times : list-like or np.1darray
            The discrete "time" points where to interpolate the function.

        Returns
        -------
        PInter : np.2darray(M, n)
            The interpolation matrix, with :math:`M` rows (size of the **times**
            parameter) and :math:`n` columns.

        """
        # Compute difference between times and Lagrange points
        times = np.asarray(times)
        with np.errstate(divide='ignore'):
            iDiff = 1 / (times[:, None] - self.points[None, :])

        # Find evaluated positions that coincide with one Lagrange point
        concom = (iDiff == np.inf) | (iDiff == -np.inf)
        i, j = np.where(concom)

        # Replace iDiff by on on those lines to get a simple copy of the value
        iDiff[i, :] = concom[i, :]

        # Compute interpolation matrix using weights
        PInter = iDiff * self.weights
        PInter /= PInter.sum(axis=-1)[:, None]

        return PInter

    def getIntegrationMatrix(self, intervals, numQuad='LEGENDRE_NUMPY'):
        r"""
        Compute the integration matrix for a given set of intervals.

        For instance, if we note :math:`\vec{f}` the vector containing the
        :math:`f_j=f(t_j)` values, and
        :math:`(\tau_{m,left}, \tau_{m,right})_{0\leq m<M}` the different
        interval where the function should be integrated using the barycentric
        interpolant polynomial.
        Then :math:`\Delta[\vec{f}]`, the vector containing the approximations
        of

        .. math::
            \int_{\tau_{m,left}}^{\tau_{m,right}} f(t)dt,

        can be obtained using :

        .. math::
            \Delta[\vec{f}] = P_{Integ} \vec{f},

        where :math:`P_{Integ}` is the interpolation matrix returned by this
        method.

        Parameters
        ----------
        intervals : list of pairs
            A list of all integration intervals.
        numQuad : str, optional
            Quadrature rule used to integrate the interpolant barycentric
            polynomial. Can be :

            - 'LEGENDRE_NUMPY' : Gauss-Legendre rule from Numpy
            - 'LEGENDRE_SCIPY' : Gauss-Legendre rule from Scipy
            - 'FEJER' : internaly implemented Fejer-I rule

            The default is 'LEGENDRE_NUMPY'.

        Returns
        -------
        PInter : np.2darray(M, n)
            The integration matrix, with :math:`M` rows (number of intervals)
            and :math:`n` columns.
        """
        if numQuad == 'LEGENDRE_NUMPY':
            # Legendre gauss rule, integrate exactly polynomials of deg. (2n-1)
            iNodes, iWeights = np.polynomial.legendre.leggauss((self.n + 1) // 2)
        elif numQuad == 'LEGENDRE_SCIPY':
            # Using Legendre scipy implementation
            iNodes, iWeights = roots_legendre((self.n + 1) // 2)
        elif numQuad == 'FEJER':
            # Fejer-I rule, integrate exactly polynomial of deg. n-1
            iNodes, iWeights = computeFejerRule(self.n - ((self.n + 1) % 2))
        else:
            raise NotImplementedError(f'numQuad={numQuad}')

        # Compute quadrature nodes for each interval
        intervals = np.array(intervals)
        aj, bj = intervals[:, 0][:, None], intervals[:, 1][:, None]
        tau, omega = iNodes[None, :], iWeights[None, :]
        tEval = (bj - aj) / 2 * tau + (bj + aj) / 2

        # Compute the integrand function on nodes
        integrand = self.getInterpolationMatrix(tEval.ravel()).T.reshape((-1,) + tEval.shape)

        # Apply quadrature rule to integrate
        integrand *= omega
        integrand *= (bj - aj) / 2
        PInter = integrand.sum(axis=-1).T

        return PInter
