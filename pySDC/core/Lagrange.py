import numpy as np


def computeFejerRule(n):
    # Fejer rule of the first kind
    # Computation using DFT (Waldvogel 2006)
    # Inspired from quadpy (https://github.com/nschloe/quadpy @Nico_Schlömer)

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
    v0 = np.concatenate([
        2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K**2),
        np.zeros(lN + 1)])
    # -- Build v1 from v0
    v1 = np.empty(len(v0) - 1, dtype=complex)
    np.conjugate(v0[:0:-1], out=v1)
    v1 += v0[:-1]
    # -- Compute inverse fourier transform
    w = np.fft.ifft(v1)
    if max(w.imag) > 1.0e-15:
        raise ValueError(
            f'Max imaginary value to important for ifft: {max(w.imag)}')
    # -- Store weights
    weights[:] = w.real

    return nodes, weights


class LagrangeApproximation(object):

    def __init__(self, points, weightComputation='STABLE', scaleRef='MAX'):

        points = np.asarray(points).ravel()

        diffs = points[:, None] - points[None, :]
        diffs[np.diag_indices_from(diffs)] = 1

        def analytic(diffs):
            # Fast implementation (unstable for large number of points)
            diffs *= 4 / (points.max() - points.min())
            invProd = np.prod(diffs, axis=1)
            invProd **= -1
            return invProd

        def logScale(diffs):
            # Stable implementation for large number of points (more expensive)
            sign = np.sign(diffs).prod(axis=1)
            wLog = -np.log(np.abs(diffs)).sum(axis=1)
            if scaleRef == 'ZERO':
                wScale = wLog[np.argmin(np.abs(points))]
            elif scaleRef == 'MAX':
                wScale = wLog.max()
            else:
                raise NotImplementedError(f'scaleRef={scaleRef}')
            invProd = np.exp(wLog - wScale)
            invProd *= sign
            return invProd

        def chebfun(diffs):
            # Implementation used in chebfun (stable for many points)
            diffs *= 4 / (points.max() - points.min())
            sign = np.sign(diffs).prod(axis=1)
            vv = np.exp(np.log(np.abs(diffs)).sum(axis=1))
            invProd = (sign * vv)
            invProd **= -1
            invProd /= np.linalg.norm(invProd, np.inf)
            return invProd

        if weightComputation == 'AUTO':
            with np.errstate(divide='raise', over='ignore'):
                try:
                    invProd = analytic(diffs)
                except FloatingPointError:
                    invProd = logScale(diffs)
        elif weightComputation == 'FAST':
            invProd = analytic(diffs)
        elif weightComputation == 'STABLE':
            invProd = logScale(diffs)
        elif weightComputation == 'CHEBFUN':
            invProd = chebfun(diffs)
        else:
            raise NotImplementedError(
                f'weightComputation={weightComputation}')
        weights = invProd

        # Store attributes
        self.points = points
        self.weights = weights
        self.weightComputation = weightComputation

    @property
    def n(self):
        return self.points.size

    def getInterpolationMatrix(self, times):

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

    def getIntegrationMatrix(self, intervals, numQuad='LEGENDRE'):

        if numQuad == 'LEGENDRE':
            # Legendre gauss rule, integrate exactly polynomials of deg. (2n-1)
            iNodes, iWeights = np.polynomial.legendre.leggauss(self.n // 2)
        elif numQuad == 'FEJER':
            # Fejer-I rule, integrate exactly polynomial of deg. n-1
            iNodes, iWeights = computeFejerRule(self.n - 1 - (self.n % 2))
        else:
            raise NotImplementedError(f'numQuad={numQuad}')

        # Compute quadrature nodes for each interval
        intervals = np.array(intervals)
        aj, bj = intervals[:, 0][:, None], intervals[:, 1][:, None]
        tau, omega = iNodes[None, :], iWeights[None, :]
        tEval = (bj - aj) / 2 * tau + (bj + aj) / 2

        # Compute the integrand function on nodes
        integrand = self.getInterpolationMatrix(tEval.ravel()).T.reshape(
            (-1,) + tEval.shape)

        # Apply quadrature rule to integrate
        integrand *= omega
        integrand *= (bj - aj) / 2
        PInter = integrand.sum(axis=-1).T

        return PInter
