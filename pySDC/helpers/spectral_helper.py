import numpy as np
import scipy
from pySDC.implementations.datatype_classes.mesh import mesh
from scipy.special import factorial
from functools import partial, wraps
import logging


def cache(func):
    """
    Decorator for caching return values of functions.
    This is very similar to `functools.cache`, but without the memory leaks (see
    https://docs.astral.sh/ruff/rules/cached-instance-method/).

    Example:

    .. code-block:: python

        num_calls = 0

        @cache
        def increment(x):
            num_calls += 1
            return x + 1

        increment(0)  # returns 1, num_calls = 1
        increment(1)  # returns 2, num_calls = 2
        increment(0)  # returns 1, num_calls = 2


    Args:
        func (function): The function you want to cache the return value of

    Returns:
        return value of func
    """
    attr_cache = f"_{func.__name__}_cache"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, attr_cache):
            setattr(self, attr_cache, {})

        cache = getattr(self, attr_cache)

        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result

    return wrapper


class SpectralHelper1D:
    """
    Abstract base class for 1D spectral discretizations. Defines a common interface with parameters and functions that
    all bases need to have.

    When implementing new bases, please take care to use the modules that are supplied as class attributes to enable
    the code for GPUs.

    Attributes:
        N (int): Resolution
        x0 (float): Coordinate of left boundary
        x1 (float): Coordinate of right boundary
        L (float): Length of the domain
        useGPU (bool): Whether to use GPUs

    """

    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    linalg = scipy.sparse.linalg
    xp = np
    distributable = False

    def __init__(self, N, x0=None, x1=None, useGPU=False, useFFTW=False):
        """
        Constructor

        Args:
            N (int): Resolution
            x0 (float): Coordinate of left boundary
            x1 (float): Coordinate of right boundary
            useGPU (bool): Whether to use GPUs
            useFFTW (bool): Whether to use FFTW for the transforms
        """
        self.N = N
        self.x0 = x0
        self.x1 = x1
        self.L = x1 - x0
        self.useGPU = useGPU
        self.plans = {}
        self.logger = logging.getLogger(name=type(self).__name__)

        if useGPU:
            self.setup_GPU()
        else:
            self.setup_CPU(useFFTW=useFFTW)

        if useGPU and useFFTW:
            raise ValueError('Please run either on GPUs or with FFTW, not both!')

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        import cupyx.scipy.sparse as sparse_lib
        import cupyx.scipy.sparse.linalg as linalg
        import cupyx.scipy.fft as fft_lib
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

        cls.xp = cp
        cls.sparse_lib = sparse_lib
        cls.linalg = linalg
        cls.fft_lib = fft_lib

    @classmethod
    def setup_CPU(cls, useFFTW=False):
        """switch to CPU modules"""

        cls.xp = np
        cls.sparse_lib = scipy.sparse
        cls.linalg = scipy.sparse.linalg

        if useFFTW:
            from mpi4py_fft import fftw

            cls.fft_backend = 'fftw'
            cls.fft_lib = fftw
        else:
            cls.fft_backend = 'scipy'
            cls.fft_lib = scipy.fft

        cls.fft_comm_backend = 'MPI'
        cls.dtype = mesh

    def get_Id(self):
        """
        Get identity matrix

        Returns:
            sparse diagonal identity matrix
        """
        return self.sparse_lib.eye(self.N)

    def get_zero(self):
        """
        Get a matrix with all zeros of the correct size.

        Returns:
            sparse matrix with zeros everywhere
        """
        return 0 * self.get_Id()

    def get_differentiation_matrix(self):
        raise NotImplementedError()

    def get_integration_matrix(self):
        raise NotImplementedError()

    def get_wavenumbers(self):
        """
        Get the grid in spectral space
        """
        raise NotImplementedError

    def get_empty_operator_matrix(self, S, O):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Args:
            S (int): Number of components in the solution
            O (sparse matrix): Zero matrix used for initialization

        Returns:
            list of lists containing sparse zeros
        """
        return [[O for _ in range(S)] for _ in range(S)]

    def get_basis_change_matrix(self, *args, **kwargs):
        """
        Some spectral discretization change the basis during differentiation. This method can be used to transfer
        between the various bases.

        This method accepts arbitrary arguments that may not be used in order to provide an easy interface for multi-
        dimensional bases. For instance, you may combine an FFT discretization with an ultraspherical discretization.
        The FFT discretization will always be in the same base, but the ultraspherical discretization uses a different
        base for every derivative. You can then ask all bases for transfer matrices from one ultraspherical derivative
        base to the next. The FFT discretization will ignore this and return an identity while the ultraspherical
        discretization will return the desired matrix. After a Kronecker product, you get the 2D version of the matrix
        you want. This is what the `SpectralHelper` does when you call the method of the same name on it.

        Returns:
            sparse bases change matrix
        """
        return self.sparse_lib.eye(self.N)

    def get_BC(self, kind):
        """
        To facilitate boundary conditions (BCs) we use either a basis where all functions satisfy the BCs automatically,
        as is the case in FFT basis for periodic BCs, or boundary bordering. In boundary bordering, specific lines in
        the matrix are replaced by the boundary conditions as obtained by this method.

        Args:
            kind (str): The type of BC you want to implement please refer to the implementations of this method in the
            individual 1D bases for what is implemented

        Returns:
            self.xp.array: Boundary condition
        """
        raise NotImplementedError(f'No boundary conditions of {kind=!r} implemented!')

    def get_filter_matrix(self, kmin=0, kmax=None):
        """
        Get a bandpass filter.

        Args:
            kmin (int): Lower limit of the bandpass filter
            kmax (int): Upper limit of the bandpass filter

        Returns:
            sparse matrix
        """

        k = abs(self.get_wavenumbers())

        kmax = max(k) if kmax is None else kmax

        mask = self.xp.logical_or(k >= kmax, k < kmin)

        if self.useGPU:
            Id = self.get_Id().get()
        else:
            Id = self.get_Id()
        F = Id.tolil()
        F[:, mask] = 0
        return F.tocsc()

    def get_1dgrid(self):
        """
        Get the grid in physical space

        Returns:
            self.xp.array: Grid
        """
        raise NotImplementedError


class ChebychevHelper(SpectralHelper1D):
    """
    The Chebychev base consists of special kinds of polynomials, with the main advantage that you can easily transform
    between physical and spectral space by discrete cosine transform.
    The differentiation in the Chebychev T base is dense, but can be preconditioned to yield a differentiation operator
    that moves to Chebychev U basis during differentiation, which is sparse. When using this technique, problems need to
    be formulated in first order formulation.

    This implementation is largely based on the Dedalus paper (https://doi.org/10.1103/PhysRevResearch.2.023068).
    """

    def __init__(self, *args, x0=-1, x1=1, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            x0 (float): Coordinate of left boundary. Note that only -1 is currently implented
            x1 (float): Coordinate of right boundary. Note that only +1 is currently implented
        """
        # need linear transformation y = ax + b with a = (x1-x0)/2 and b = (x1+x0)/2
        self.lin_trf_fac = (x1 - x0) / 2
        self.lin_trf_off = (x1 + x0) / 2
        super().__init__(*args, x0=x0, x1=x1, **kwargs)

        self.norm = self.get_norm()

    def get_1dgrid(self):
        '''
        Generates a 1D grid with Chebychev points. These are clustered at the boundary. You need this kind of grid to
        use discrete cosine transformation (DCT) to get the Chebychev representation. If you want a different grid, you
        need to do an affine transformation before any Chebychev business.

        Returns:
            numpy.ndarray: 1D grid
        '''
        return self.lin_trf_fac * self.xp.cos(np.pi / self.N * (self.xp.arange(self.N) + 0.5)) + self.lin_trf_off

    def get_wavenumbers(self):
        """Get the domain in spectral space"""
        return self.xp.arange(self.N)

    @cache
    def get_conv(self, name, N=None):
        '''
        Get conversion matrix between different kinds of polynomials. The supported kinds are
         - T: Chebychev polynomials of first kind
         - U: Chebychev polynomials of second kind
         - D: Dirichlet recombination.

        You get the desired matrix by choosing a name as ``A2B``. I.e. ``T2U`` for the conversion matrix from T to U.
        Once generates matrices are cached. So feel free to call the method as often as you like.

        Args:
         name (str): Conversion code, e.g. 'T2U'
         N (int): Size of the matrix (optional)

        Returns:
            scipy.sparse: Sparse conversion matrix
        '''
        N = N if N else self.N
        sp = self.sparse_lib
        xp = self.xp

        def get_forward_conv(name):
            if name == 'T2U':
                mat = (sp.eye(N) - sp.diags(xp.ones(N - 2), offsets=+2)).tocsc() / 2.0
                mat[:, 0] *= 2
            elif name == 'D2T':
                mat = sp.eye(N) - sp.diags(xp.ones(N - 2), offsets=+2)
            elif name[0] == name[-1]:
                mat = self.sparse_lib.eye(self.N)
            else:
                raise NotImplementedError(f'Don\'t have conversion matrix {name!r}')
            return mat

        try:
            mat = get_forward_conv(name)
        except NotImplementedError as E:
            try:
                fwd = get_forward_conv(name[::-1])
                import scipy.sparse as sp

                if self.sparse_lib == sp:
                    mat = self.sparse_lib.linalg.inv(fwd.tocsc())
                else:
                    mat = self.sparse_lib.csc_matrix(sp.linalg.inv(fwd.tocsc().get()))
            except NotImplementedError:
                raise NotImplementedError from E

        return mat

    def get_basis_change_matrix(self, conv='T2T', **kwargs):
        """
        As the differentiation matrix in Chebychev-T base is dense but is sparse when simultaneously changing base to
        Chebychev-U, you may need a basis change matrix to transfer the other matrices as well. This function returns a
        conversion matrix from `ChebychevHelper.get_conv`. Not that `**kwargs` are used to absorb arguments for other
        bases, see documentation of `SpectralHelper1D.get_basis_change_matrix`.

        Args:
            conv (str): Conversion code, i.e. T2U

        Returns:
            Sparse conversion matrix
        """
        return self.get_conv(conv)

    def get_integration_matrix(self, lbnd=0):
        """
        Get matrix for integration

        Args:
            lbnd (float): Lower bound for integration, only 0 is currently implemented

        Returns:
           Sparse integration matrix
        """
        S = self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1) @ self.get_conv('T2U')
        n = self.xp.arange(self.N)
        if lbnd == 0:
            S = S.tocsc()
            S[0, 1::2] = (
                (n / (2 * (self.xp.arange(self.N) + 1)))[1::2]
                * (-1) ** (self.xp.arange(self.N // 2))
                / (np.append([1], self.xp.arange(self.N // 2 - 1) + 1))
            ) * self.lin_trf_fac
        else:
            raise NotImplementedError(f'This function allows to integrate only from x=0, you attempted from x={lbnd}.')
        return S

    def get_differentiation_matrix(self, p=1):
        '''
        Keep in mind that the T2T differentiation matrix is dense.

        Args:
            p (int): Derivative you want to compute

        Returns:
            numpy.ndarray: Differentiation matrix
        '''
        D = self.xp.zeros((self.N, self.N))
        for j in range(self.N):
            for k in range(j):
                D[k, j] = 2 * j * ((j - k) % 2)

        D[0, :] /= 2
        return self.sparse_lib.csc_matrix(self.xp.linalg.matrix_power(D, p)) / self.lin_trf_fac**p

    @cache
    def get_norm(self, N=None):
        '''
        Get normalization for converting Chebychev coefficients and DCT

        Args:
            N (int, optional): Resolution

        Returns:
            self.xp.array: Normalization
        '''
        N = self.N if N is None else N
        norm = self.xp.ones(N) / N
        norm[0] /= 2
        return norm

    def transform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        DCT along axes. `kwargs` will be passed on to the FFT library.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform along

        Returns:
            Data in spectral space
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        kwargs['norm'] = kwargs.get('norm', 'backward')

        trf = self.fft_lib.dctn(u, *args, axes=axes, type=2, **kwargs)
        for axis in axes:

            if self.N < trf.shape[axis]:
                # mpi4py-fft implements padding only for FFT, where the frequencies are sorted such that the zeros are
                # removed in the middle rather than the end. We need to resort this here and put the highest frequencies
                # in the middle.
                _trf = self.xp.zeros_like(trf)
                N = self.N
                N_pad = _trf.shape[axis] - N
                end_first_half = N // 2 + 1

                # copy first "half"
                su = [slice(None)] * trf.ndim
                su[axis] = slice(0, end_first_half)
                _trf[tuple(su)] = trf[tuple(su)]

                # copy second "half"
                su = [slice(None)] * u.ndim
                su[axis] = slice(end_first_half + N_pad, None)
                s_u = [slice(None)] * u.ndim
                s_u[axis] = slice(end_first_half, N)
                _trf[tuple(su)] = trf[tuple(s_u)]

                # # copy values to be cut
                # su = [slice(None)] * u.ndim
                # su[axis] = slice(end_first_half, end_first_half + N_pad)
                # s_u = [slice(None)] * u.ndim
                # s_u[axis] = slice(-N_pad, None)
                # _trf[tuple(su)] = trf[tuple(s_u)]

                trf = _trf

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)
            norm = self.xp.ones(trf.shape[axis]) * self.norm[-1]
            norm[: self.N] = self.norm
            trf *= norm[(*expansion,)]
        return trf

    def itransform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        Inverse DCT along axis.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform along

        Returns:
            Data in physical space
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        kwargs['norm'] = kwargs.get('norm', 'backward')
        kwargs['overwrite_x'] = kwargs.get('overwrite_x', False)

        for axis in axes:

            if self.N == u.shape[axis]:
                _u = u.copy()
            else:
                # mpi4py-fft implements padding only for FFT, where the frequencies are sorted such that the zeros are
                # added in the middle rather than the end. We need to resort this here and put the padding in the end.
                N = self.N
                _u = self.xp.zeros_like(u)

                # copy first half
                su = [slice(None)] * u.ndim
                su[axis] = slice(0, N // 2 + 1)
                _u[tuple(su)] = u[tuple(su)]

                # copy second half
                su = [slice(None)] * u.ndim
                su[axis] = slice(-(N // 2), None)
                s_u = [slice(None)] * u.ndim
                s_u[axis] = slice(N // 2, N // 2 + (N // 2))
                _u[tuple(s_u)] = u[tuple(su)]

                if N % 2 == 0:
                    su = [slice(None)] * u.ndim
                    su[axis] = N // 2
                    _u[tuple(su)] *= 2

            # generate norm
            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)
            norm = self.xp.ones(_u.shape[axis])
            norm[: self.N] = self.norm
            norm = self.get_norm(u.shape[axis]) * _u.shape[axis] / self.N

            _u /= norm[(*expansion,)]

        return self.fft_lib.idctn(_u, *args, axes=axes, type=2, **kwargs)

    def get_BC(self, kind, **kwargs):
        """
        Get boundary condition row for boundary bordering. `kwargs` will be passed on to implementations of the BC of
        the kind you choose. Specifically, `x` for `'dirichlet'` boundary condition, which is the coordinate at which to
        set the BC.

        Args:
            kind ('integral' or 'dirichlet'): Kind of boundary condition you want
        """
        if kind.lower() == 'integral':
            return self.get_integ_BC_row(**kwargs)
        elif kind.lower() == 'dirichlet':
            return self.get_Dirichlet_BC_row(**kwargs)
        else:
            return super().get_BC(kind)

    def get_integ_BC_row(self):
        """
        Get a row for generating integral BCs with T polynomials.
        It returns the values of the integrals of T polynomials over the entire interval.

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        n = self.xp.arange(self.N) + 1
        me = self.xp.zeros_like(n).astype(float)
        me[2:] = ((-1) ** n[1:-1] + 1) / (1 - n[1:-1] ** 2)
        me[0] = 2.0
        return me

    def get_Dirichlet_BC_row(self, x):
        """
        Get a row for generating Dirichlet BCs at x with T polynomials.
        It returns the values of the T polynomials at x.

        Args:
            x (float): Position of the boundary condition

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        if x == -1:
            return (-1) ** self.xp.arange(self.N)
        elif x == 1:
            return self.xp.ones(self.N)
        elif x == 0:
            n = (1 + (-1) ** self.xp.arange(self.N)) / 2
            n[2::4] *= -1
            return n
        else:
            raise NotImplementedError(f'Don\'t know how to generate Dirichlet BC\'s at {x=}!')

    def get_Dirichlet_recombination_matrix(self):
        '''
        Get matrix for Dirichlet recombination, which changes the basis to have sparse boundary conditions.
        This makes for a good right preconditioner.

        Returns:
            scipy.sparse: Sparse conversion matrix
        '''
        N = self.N
        sp = self.sparse_lib
        xp = self.xp

        return sp.eye(N) - sp.diags(xp.ones(N - 2), offsets=+2)


class UltrasphericalHelper(ChebychevHelper):
    """
    This implementation follows https://doi.org/10.1137/120865458.
    The ultraspherical method works in Chebychev polynomials as well, but also uses various Gegenbauer polynomials.
    The idea is that for every derivative of Chebychev T polynomials, there is a basis of Gegenbauer polynomials where the differentiation matrix is a single off-diagonal.
    There are also conversion operators from one derivative basis to the next that are sparse.

    This basis is used like this: For every equation that you have, look for the highest derivative and bump all matrices to the correct basis. If your highest derivative is 2 and you have an identity, it needs to get bumped from 0 to 1 and from 1 to 2. If you have a first derivative as well, it needs to be bumped from 1 to 2.
    You don't need the same resulting basis in all equations. You just need to take care that you translate the right hand side to the correct basis as well.
    """

    def get_differentiation_matrix(self, p=1):
        """
        Notice that while sparse, this matrix is not diagonal, which means the inversion cannot be parallelized easily.

        Args:
            p (int): Order of the derivative

        Returns:
            sparse differentiation matrix
        """
        sp = self.sparse_lib
        xp = self.xp
        N = self.N
        l = p
        return 2 ** (l - 1) * factorial(l - 1) * sp.diags(xp.arange(N - l) + l, offsets=l) / self.lin_trf_fac**p

    def get_S(self, lmbda):
        """
        Get matrix for bumping the derivative base by one from lmbda to lmbda + 1. This is the same language as in
        https://doi.org/10.1137/120865458.

        Args:
            lmbda (int): Ingoing derivative base

        Returns:
            sparse matrix: Conversion from derivative base lmbda to lmbda + 1
        """
        N = self.N

        if lmbda == 0:
            sp = scipy.sparse
            mat = ((sp.eye(N) - sp.diags(np.ones(N - 2), offsets=+2)) / 2.0).tolil()
            mat[:, 0] *= 2
        else:
            sp = self.sparse_lib
            xp = self.xp
            mat = sp.diags(lmbda / (lmbda + xp.arange(N))) - sp.diags(
                lmbda / (lmbda + 2 + xp.arange(N - 2)), offsets=+2
            )

        return self.sparse_lib.csc_matrix(mat)

    def get_basis_change_matrix(self, p_in=0, p_out=0, **kwargs):
        """
        Get a conversion matrix from derivative base `p_in` to `p_out`.

        Args:
            p_out (int): Resulting derivative base
            p_in (int): Ingoing derivative base
        """
        mat_fwd = self.sparse_lib.eye(self.N)
        for i in range(min([p_in, p_out]), max([p_in, p_out])):
            mat_fwd = self.get_S(i) @ mat_fwd

        if p_out > p_in:
            return mat_fwd

        else:
            # We have to invert the matrix on CPU because the GPU equivalent is not implemented in CuPy at the time of writing.
            import scipy.sparse as sp

            if self.useGPU:
                mat_fwd = mat_fwd.get()

            mat_bck = sp.linalg.inv(mat_fwd.tocsc())

            return self.sparse_lib.csc_matrix(mat_bck)

    def get_integration_matrix(self):
        """
        Get an integration matrix. Please use `UltrasphericalHelper.get_integration_constant` afterwards to compute the
        integration constant such that integration starts from x=-1.

        Example:

        .. code-block:: python

            import numpy as np
            from pySDC.helpers.spectral_helper import UltrasphericalHelper

            N = 4
            helper = UltrasphericalHelper(N)
            coeffs = np.random.random(N)
            coeffs[-1] = 0

            poly = np.polynomial.Chebyshev(coeffs)

            S = helper.get_integration_matrix()
            U_hat = S @ coeffs
            U_hat[0] = helper.get_integration_constant(U_hat, axis=-1)

            assert np.allclose(poly.integ(lbnd=-1).coef[:-1], U_hat)

        Returns:
            sparse integration matrix
        """
        return (
            self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1)
            @ self.get_basis_change_matrix(p_out=1, p_in=0)
            * self.lin_trf_fac
        )

    def get_integration_constant(self, u_hat, axis):
        """
        Get integration constant for lower bound of -1. See documentation of `UltrasphericalHelper.get_integration_matrix` for details.

        Args:
            u_hat: Solution in spectral space
            axis: Axis you want to integrate over

        Returns:
            Integration constant, has one less dimension than `u_hat`
        """
        slices = [
            None,
        ] * u_hat.ndim
        slices[axis] = slice(1, u_hat.shape[axis])
        return self.xp.sum(u_hat[(*slices,)] * (-1) ** (self.xp.arange(u_hat.shape[axis] - 1)), axis=axis)


class FFTHelper(SpectralHelper1D):
    distributable = True

    def __init__(self, *args, x0=0, x1=2 * np.pi, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            x0 (float, optional): Coordinate of left boundary
            x1 (float, optional): Coordinate of right boundary
        """
        super().__init__(*args, x0=x0, x1=x1, **kwargs)

    def get_1dgrid(self):
        """
        We use equally spaced points including the left boundary and not including the right one, which is the left boundary.
        """
        dx = self.L / self.N
        return self.xp.arange(self.N) * dx + self.x0

    def get_wavenumbers(self):
        """
        Be careful that this ordering is very unintuitive.
        """
        return self.xp.fft.fftfreq(self.N, 1.0 / self.N) * 2 * np.pi / self.L

    def get_differentiation_matrix(self, p=1):
        """
        This matrix is diagonal, allowing to invert concurrently.

        Args:
            p (int): Order of the derivative

        Returns:
            sparse differentiation matrix
        """
        k = self.get_wavenumbers()

        if self.useGPU:
            if p > 1:
                # Have to raise the matrix to power p on CPU because the GPU equivalent is not implemented in CuPy at the time of writing.
                from scipy.sparse.linalg import matrix_power

                D = self.sparse_lib.diags(1j * k).get()
                return self.sparse_lib.csc_matrix(matrix_power(D, p))
            else:
                return self.sparse_lib.diags(1j * k)
        else:
            return self.linalg.matrix_power(self.sparse_lib.diags(1j * k), p)

    def get_integration_matrix(self, p=1):
        """
        Get integration matrix to compute `p`-th integral over the entire domain.

        Args:
            p (int): Order of integral you want to compute

        Returns:
            sparse integration matrix
        """
        k = self.xp.array(self.get_wavenumbers(), dtype='complex128')
        k[0] = 1j * self.L
        return self.linalg.matrix_power(self.sparse_lib.diags(1 / (1j * k)), p)

    def get_plan(self, u, forward, *args, **kwargs):
        if self.fft_lib.__name__ == 'mpi4py_fft.fftw':
            if 'axes' in kwargs.keys():
                kwargs['axes'] = tuple(kwargs['axes'])
            key = (forward, u.shape, args, *(me for me in kwargs.values()))
            if key in self.plans.keys():
                return self.plans[key]
            else:
                self.logger.debug(f'Generating FFT plan for {key=}')
                transform = self.fft_lib.fftn(u, *args, **kwargs) if forward else self.fft_lib.ifftn(u, *args, **kwargs)
                self.plans[key] = transform

            return self.plans[key]
        else:
            if forward:
                return partial(self.fft_lib.fftn, norm=kwargs.get('norm', 'backward'))
            else:
                return partial(self.fft_lib.ifftn, norm=kwargs.get('norm', 'forward'))

    def transform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        FFT along axes. `kwargs` are passed on to the FFT library.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform over

        Returns:
            transformed data
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        plan = self.get_plan(u, *args, forward=True, axes=axes, **kwargs)
        return plan(u, *args, axes=axes, **kwargs)

    def itransform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        Inverse FFT.

        Args:
            u: Data you want to transform
            axes (tuple): Axes over which to transform

        Returns:
            transformed data
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        plan = self.get_plan(u, *args, forward=False, axes=axes, **kwargs)
        return plan(u, *args, axes=axes, **kwargs) / np.prod([u.shape[axis] for axis in axes])

    def get_BC(self, kind):
        """
        Get a sort of boundary condition. You can use `kind=integral`, to fix the integral, or you can use `kind=Nyquist`.
        The latter is not really a boundary condition, but is used to set the Nyquist mode to some value, preferably zero.
        You should set the Nyquist mode zero when the solution in physical space is real and the resolution is even.

        Args:
            kind ('integral' or 'nyquist'): Kind of BC

        Returns:
            self.xp.ndarray: Boundary condition row
        """
        if kind.lower() == 'integral':
            return self.get_integ_BC_row()
        elif kind.lower() == 'nyquist':
            assert (
                self.N % 2 == 0
            ), f'Do not eliminate the Nyquist mode with odd resolution as it is fully resolved. You chose {self.N} in this axis'
            BC = self.xp.zeros(self.N)
            BC[self.get_Nyquist_mode_index()] = 1
            return BC
        else:
            return super().get_BC(kind)

    def get_Nyquist_mode_index(self):
        """
        Compute the index of the Nyquist mode, i.e. the mode with the lowest wavenumber, which doesn't have a positive
        counterpart for even resolution. This means real waves of this wave number cannot be properly resolved and you
        are best advised to set this mode zero if representing real functions on even-resolution grids is what you're
        after.

        Returns:
            int: Index of the Nyquist mode
        """
        k = self.get_wavenumbers()
        Nyquist_mode = min(k)
        return self.xp.where(k == Nyquist_mode)[0][0]

    def get_integ_BC_row(self):
        """
        Only the 0-mode has non-zero integral with FFT basis in periodic BCs
        """
        me = self.xp.zeros(self.N)
        me[0] = self.L / self.N
        return me


class SpectralHelper:
    """
    This class has three functions:
      - Easily assemble matrices containing multiple equations
      - Direct product of 1D bases to solve problems in more dimensions
      - Distribute the FFTs to facilitate concurrency.

    Attributes:
        comm (mpi4py.Intracomm): MPI communicator
        debug (bool): Perform additional checks at extra computational cost
        useGPU (bool): Whether to use GPUs
        axes (list): List of 1D bases
        components (list): List of strings of the names of components in the equations
        full_BCs (list): List of Dictionaries containing all information about the boundary conditions
        BC_mat (list): List of lists of sparse matrices to put BCs into and eventually assemble the BC matrix from
        BCs (sparse matrix): Matrix containing only the BCs
        fft_cache (dict): Cache FFTs of various shapes here to facilitate padding and so on
        BC_rhs_mask (self.xp.ndarray): Mask values that contain boundary conditions in the right hand side
        BC_zero_index (self.xp.ndarray): Indeces of rows in the matrix that are replaced by BCs
        BC_line_zero_matrix (sparse matrix): Matrix that zeros rows where we can then add the BCs in using `BCs`
        rhs_BCs_hat (self.xp.ndarray): Boundary conditions in spectral space
        global_shape (tuple): Global shape of the solution as in `mpi4py-fft`
        fft_obj: When using distributed FFTs, this will be a parallel transform object from `mpi4py-fft`
        init (tuple): This is the same `init` that is used throughout the problem classes
        init_forward (tuple): This is the equivalent of `init` in spectral space
    """

    xp = np
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    linalg = scipy.sparse.linalg
    dtype = mesh
    fft_backend = 'scipy'
    fft_comm_backend = 'MPI'

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        import cupyx.scipy.sparse as sparse_lib
        import cupyx.scipy.sparse.linalg as linalg
        import cupyx.scipy.fft as fft_lib
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

        cls.xp = cp
        cls.sparse_lib = sparse_lib
        cls.linalg = linalg

        cls.fft_lib = fft_lib
        cls.fft_backend = 'cupyx-scipy'
        cls.fft_comm_backend = 'NCCL'

        cls.dtype = cupy_mesh

    @classmethod
    def setup_CPU(cls, useFFTW=False):
        """switch to CPU modules"""

        cls.xp = np
        cls.sparse_lib = scipy.sparse
        cls.linalg = scipy.sparse.linalg

        if useFFTW:
            from mpi4py_fft import fftw

            cls.fft_backend = 'fftw'
            cls.fft_lib = fftw
        else:
            cls.fft_backend = 'scipy'
            cls.fft_lib = scipy.fft

        cls.fft_comm_backend = 'MPI'
        cls.dtype = mesh

    def __init__(self, comm=None, useGPU=False, debug=False):
        """
        Constructor

        Args:
            comm (mpi4py.Intracomm): MPI communicator
            useGPU (bool): Whether to use GPUs
            debug (bool): Perform additional checks at extra computational cost
        """
        self.comm = comm
        self.debug = debug
        self.useGPU = useGPU

        if useGPU:
            self.setup_GPU()
        else:
            self.setup_CPU()

        self.axes = []
        self.components = []

        self.full_BCs = []
        self.BC_mat = None
        self.BCs = None

        self.fft_cache = {}
        self.fft_dealias_shape_cache = {}

        self.logger = logging.getLogger(name='Spectral Discretization')
        if debug:
            self.logger.setLevel(logging.DEBUG)

    @property
    def u_init(self):
        """
        Get empty data container in physical space
        """
        return self.dtype(self.init)

    @property
    def u_init_forward(self):
        """
        Get empty data container in spectral space
        """
        return self.dtype(self.init_forward)

    @property
    def u_init_physical(self):
        """
        Get empty data container in physical space
        """
        return self.dtype(self.init_physical)

    @property
    def shape(self):
        """
        Get shape of individual solution component
        """
        return self.init[0][1:]

    @property
    def ndim(self):
        return len(self.axes)

    @property
    def ncomponents(self):
        return len(self.components)

    @property
    def V(self):
        """
        Get domain volume
        """
        return np.prod([me.L for me in self.axes])

    def add_axis(self, base, *args, **kwargs):
        """
        Add an axis to the domain by deciding on suitable 1D base.
        Arguments to the bases are forwarded using `*args` and `**kwargs`. Please refer to the documentation of the 1D
        bases for possible arguments.

        Args:
            base (str): 1D spectral method
        """
        kwargs['useGPU'] = self.useGPU

        if base.lower() in ['chebychov', 'chebychev', 'cheby', 'chebychovhelper']:
            self.axes.append(ChebychevHelper(*args, **kwargs))
        elif base.lower() in ['fft', 'fourier', 'ffthelper']:
            self.axes.append(FFTHelper(*args, **kwargs))
        elif base.lower() in ['ultraspherical', 'gegenbauer']:
            self.axes.append(UltrasphericalHelper(*args, **kwargs))
        else:
            raise NotImplementedError(f'{base=!r} is not implemented!')
        self.axes[-1].xp = self.xp
        self.axes[-1].sparse_lib = self.sparse_lib

    def add_component(self, name):
        """
        Add solution component(s).

        Args:
            name (str or list of strings): Name(s) of component(s)
        """
        if type(name) in [list, tuple]:
            for me in name:
                self.add_component(me)
        elif type(name) in [str]:
            if name in self.components:
                raise Exception(f'{name=!r} is already added to this problem!')
            self.components.append(name)
        else:
            raise NotImplementedError

    def index(self, name):
        """
        Get the index of component `name`.

        Args:
            name (str or list of strings): Name(s) of component(s)

        Returns:
            int: Index of the component
        """
        if type(name) in [str, int]:
            return self.components.index(name)
        elif type(name) in [list, tuple]:
            return (self.index(me) for me in name)
        else:
            raise NotImplementedError(f'Don\'t know how to compute index for {type(name)=}')

    def get_empty_operator_matrix(self, diag=False):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Args:
            diag (bool): Whether operator is block-diagonal

        Returns:
            list containing sparse zeros
        """
        S = len(self.components)
        O = self.get_Id() * 0
        if diag:
            return [O for _ in range(S)]
        else:
            return [[O for _ in range(S)] for _ in range(S)]

    def get_BC(self, axis, kind, line=-1, scalar=False, **kwargs):
        """
        Use this method for boundary bordering. It gets the respective matrix row and embeds it into a matrix.
        Pay attention that if you have multiple BCs in a single equation, you need to put them in different lines.
        Typically, the last line that does not contain a BC is the best choice.
        Forward arguments for the boundary conditions using `kwargs`. Refer to documentation of 1D bases for details.

        Args:
            axis (int): Axis you want to add the BC to
            kind (str): kind of BC, e.g. Dirichlet
            line (int): Line you want the BC to go in
            scalar (bool): Put the BC in all space positions in the other direction

        Returns:
            sparse matrix containing the BC
        """
        sp = scipy.sparse

        base = self.axes[axis]

        BC = sp.eye(base.N).tolil() * 0
        if self.useGPU:
            BC[line, :] = base.get_BC(kind=kind, **kwargs).get()
        else:
            BC[line, :] = base.get_BC(kind=kind, **kwargs)

        ndim = len(self.axes)
        if ndim == 1:
            mat = self.sparse_lib.csc_matrix(BC)
        elif ndim == 2:
            axis2 = (axis + 1) % ndim

            if scalar:
                _Id = self.sparse_lib.diags(self.xp.append([1], self.xp.zeros(self.axes[axis2].N - 1)))
            else:
                _Id = self.axes[axis2].get_Id()

            Id = self.get_local_slice_of_1D_matrix(self.axes[axis2].get_Id() @ _Id, axis=axis2)

            mats = [
                None,
            ] * ndim
            mats[axis] = self.get_local_slice_of_1D_matrix(BC, axis=axis)
            mats[axis2] = Id
            mat = self.sparse_lib.csc_matrix(self.sparse_lib.kron(*mats))
        elif ndim == 3:
            mats = [
                None,
            ] * ndim

            for ax in range(ndim):
                if ax == axis:
                    continue

                if scalar:
                    _Id = self.sparse_lib.diags(self.xp.append([1], self.xp.zeros(self.axes[ax].N - 1)))
                else:
                    _Id = self.axes[ax].get_Id()

                mats[ax] = self.get_local_slice_of_1D_matrix(self.axes[ax].get_Id() @ _Id, axis=ax)

            mats[axis] = self.get_local_slice_of_1D_matrix(BC, axis=axis)

            mat = self.sparse_lib.csc_matrix(self.sparse_lib.kron(mats[0], self.sparse_lib.kron(*mats[1:])))
        else:
            raise NotImplementedError(
                f'Matrix expansion for boundary conditions not implemented for {ndim} dimensions!'
            )
        mat = self.eliminate_zeros(mat)
        return mat

    def remove_BC(self, component, equation, axis, kind, line=-1, scalar=False, **kwargs):
        """
        Remove a BC from the matrix. This is useful e.g. when you add a non-scalar BC and then need to selectively
        remove single BCs again, as in incompressible Navier-Stokes, for instance.
        Forwards arguments for the boundary conditions using `kwargs`. Refer to documentation of 1D bases for details.

        Args:
            component (str): Name of the component the BC should act on
            equation (str): Name of the equation for the component you want to put the BC in
            axis (int): Axis you want to add the BC to
            kind (str): kind of BC, e.g. Dirichlet
            v: Value of the BC
            line (int): Line you want the BC to go in
            scalar (bool): Put the BC in all space positions in the other direction
        """
        _BC = self.get_BC(axis=axis, kind=kind, line=line, scalar=scalar, **kwargs)
        _BC = self.eliminate_zeros(_BC)
        self.BC_mat[self.index(equation)][self.index(component)] -= _BC

        if scalar:
            slices = [self.index(equation)] + [
                0,
            ] * self.ndim
            slices[axis + 1] = line
        else:
            slices = (
                [self.index(equation)]
                + [slice(0, self.init[0][i + 1]) for i in range(axis)]
                + [line]
                + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
            )
        N = self.axes[axis].N
        if (N + line) % N in self.xp.arange(N)[self.local_slice()[axis]]:
            self.BC_rhs_mask[(*slices,)] = False

    def add_BC(self, component, equation, axis, kind, v, line=-1, scalar=False, **kwargs):
        """
        Add a BC to the matrix. Note that you need to convert the list of lists of BCs that this method generates to a
        single sparse matrix by calling `setup_BCs` after adding/removing all BCs.
        Forward arguments for the boundary conditions using `kwargs`. Refer to documentation of 1D bases for details.

        Args:
            component (str): Name of the component the BC should act on
            equation (str): Name of the equation for the component you want to put the BC in
            axis (int): Axis you want to add the BC to
            kind (str): kind of BC, e.g. Dirichlet
            v: Value of the BC
            line (int): Line you want the BC to go in
            scalar (bool): Put the BC in all space positions in the other direction
        """
        _BC = self.get_BC(axis=axis, kind=kind, line=line, scalar=scalar, **kwargs)
        self.BC_mat[self.index(equation)][self.index(component)] += _BC
        self.full_BCs += [
            {
                'component': component,
                'equation': equation,
                'axis': axis,
                'kind': kind,
                'v': v,
                'line': line,
                'scalar': scalar,
                **kwargs,
            }
        ]

        if scalar:
            slices = [self.index(equation)] + [
                0,
            ] * self.ndim
            slices[axis + 1] = line
            if self.comm:
                if self.comm.rank == 0:
                    self.BC_rhs_mask[(*slices,)] = True
            else:
                self.BC_rhs_mask[(*slices,)] = True
        else:
            slices = [self.index(equation), *self.global_slice(True)]
            N = self.axes[axis].N
            if (N + line) % N in self.get_indices(True)[axis]:
                slices[axis + 1] = (N + line) % N - self.local_slice()[axis].start
                self.BC_rhs_mask[(*slices,)] = True

    def setup_BCs(self):
        """
        Convert the list of lists of BCs to the boundary condition operator.
        Also, boundary bordering requires to zero out all other entries in the matrix in rows containing a boundary
        condition. This method sets up a suitable sparse matrix to do this.
        """
        sp = self.sparse_lib
        self.BCs = self.convert_operator_matrix_to_operator(self.BC_mat)
        self.BC_zero_index = self.xp.arange(np.prod(self.init[0]))[self.BC_rhs_mask.flatten()]

        diags = self.xp.ones(self.BCs.shape[0])
        diags[self.BC_zero_index] = 0
        self.BC_line_zero_matrix = sp.diags(diags)

        # prepare BCs in spectral space to easily add to the RHS
        rhs_BCs = self.put_BCs_in_rhs(self.u_init)
        self.rhs_BCs_hat = self.transform(rhs_BCs)

    def check_BCs(self, u):
        """
        Check that the solution satisfies the boundary conditions

        Args:
            u: The solution you want to check
        """
        assert self.ndim < 3
        for axis in range(self.ndim):
            BCs = [me for me in self.full_BCs if me["axis"] == axis and not me["scalar"]]

            if len(BCs) > 0:
                u_hat = self.transform(u, axes=(axis - self.ndim,))
                for BC in BCs:
                    kwargs = {
                        key: value
                        for key, value in BC.items()
                        if key not in ['component', 'equation', 'axis', 'v', 'line', 'scalar']
                    }

                    if axis == 0:
                        get = self.axes[axis].get_BC(**kwargs) @ u_hat[self.index(BC['component'])]
                    elif axis == 1:
                        get = u_hat[self.index(BC['component'])] @ self.axes[axis].get_BC(**kwargs)
                    want = BC['v']
                    assert self.xp.allclose(
                        get, want
                    ), f'Unexpected BC in {BC["component"]} in equation {BC["equation"]}, line {BC["line"]}! Got {get}, wanted {want}'

    def put_BCs_in_matrix(self, A):
        """
        Put the boundary conditions in a matrix by replacing rows with BCs.
        """
        return self.BC_line_zero_matrix @ A + self.BCs

    def put_BCs_in_rhs_hat(self, rhs_hat):
        """
        Put the BCs in the right hand side in spectral space for solving.
        This function needs no transforms and caches a mask for faster subsequent use.

        Args:
            rhs_hat: Right hand side in spectral space

        Returns:
            rhs in spectral space with BCs
        """
        if not hasattr(self, '_rhs_hat_zero_mask'):
            """
            Generate a mask where we need to set values in the rhs in spectral space to zero, such that can replace them
            by the boundary conditions. The mask is then cached.
            """
            self._rhs_hat_zero_mask = self.newDistArray().astype(bool)

            for axis in range(self.ndim):
                for bc in self.full_BCs:
                    if axis == bc['axis']:
                        slices = [self.index(bc['equation']), *self.global_slice(True)]
                        N = self.axes[axis].N
                        line = bc['line']
                        if (N + line) % N in self.get_indices(True)[axis]:
                            slices[axis + 1] = (N + line) % N - self.local_slice()[axis].start
                            self._rhs_hat_zero_mask[(*slices,)] = True

        rhs_hat[self._rhs_hat_zero_mask] = 0
        return rhs_hat + self.rhs_BCs_hat

    def put_BCs_in_rhs(self, rhs):
        """
        Put the BCs in the right hand side for solving.
        This function will transform along each axis individually and add all BCs in that axis.
        Consider `put_BCs_in_rhs_hat` to add BCs with no extra transforms needed.

        Args:
            rhs: Right hand side in physical space

        Returns:
            rhs in physical space with BCs
        """
        assert rhs.ndim > 1, 'rhs must not be flattened here!'

        ndim = self.ndim

        for axis in range(ndim):
            _rhs_hat = self.transform(rhs, axes=(axis - ndim,))

            for bc in self.full_BCs:

                if axis == bc['axis']:
                    _slice = [self.index(bc['equation']), *self.global_slice(True)]

                    N = self.axes[axis].N
                    line = bc['line']
                    if (N + line) % N in self.get_indices(True)[axis]:
                        _slice[axis + 1] = (N + line) % N - self.local_slice()[axis].start
                        _rhs_hat[(*_slice,)] = bc['v']

            rhs = self.itransform(_rhs_hat, axes=(axis - ndim,))

        return rhs

    def add_equation_lhs(self, A, equation, relations):
        """
        Add the left hand part (that you want to solve implicitly) of an equation to a list of lists of sparse matrices
        that you will convert to an operator later.

        Example:
        Setup linear operator `L` for 1D heat equation using Chebychev method in first order form and T-to-U
        preconditioning:

        .. code-block:: python
            helper = SpectralHelper()

            helper.add_axis(base='chebychev', N=8)
            helper.add_component(['u', 'ux'])
            helper.setup_fft()

            I = helper.get_Id()
            Dx = helper.get_differentiation_matrix(axes=(0,))
            T2U = helper.get_basis_change_matrix('T2U')

            L_lhs = {
                'ux': {'u': -T2U @ Dx, 'ux': T2U @ I},
                'u': {'ux': -(T2U @ Dx)},
            }

            operator = helper.get_empty_operator_matrix()
            for line, equation in L_lhs.items():
                helper.add_equation_lhs(operator, line, equation)

            L = helper.convert_operator_matrix_to_operator(operator)

        Args:
            A (list of lists of sparse matrices): The operator to be
            equation (str): The equation of the component you want this in
            relations: (dict): Relations between quantities
        """
        for k, v in relations.items():
            A[self.index(equation)][self.index(k)] = v

    def eliminate_zeros(self, A):
        """
        Eliminate zeros from sparse matrix. This can reduce memory footprint of matrices somewhat.
        Note: At the time of writing, there are memory problems in the cupy implementation of `eliminate_zeros`.
        Therefore, this function copies the matrix to host, eliminates the zeros there and then copies back to GPU.

        Args:
            A: sparse matrix to be pruned

        Returns:
            CSC sparse matrix
        """
        if self.useGPU:
            A = A.get()
        A = A.tocsc()
        A.eliminate_zeros()
        if self.useGPU:
            A = self.sparse_lib.csc_matrix(A)
        return A

    def convert_operator_matrix_to_operator(self, M):
        """
        Promote the list of lists of sparse matrices to a single sparse matrix that can be used as linear operator.
        See documentation of `SpectralHelper.add_equation_lhs` for an example.

        Args:
            M (list of lists of sparse matrices): The operator to be

        Returns:
            sparse linear operator
        """
        if len(self.components) == 1:
            op = M[0][0]
        else:
            op = self.sparse_lib.bmat(M, format='csc')

        op = self.eliminate_zeros(op)
        return op

    def get_wavenumbers(self):
        """
        Get grid in spectral space
        """
        grids = [self.axes[i].get_wavenumbers()[self.local_slice(True)[i]] for i in range(len(self.axes))]
        return self.xp.meshgrid(*grids, indexing='ij')

    def get_grid(self, forward_output=False):
        """
        Get grid in physical space
        """
        grids = [self.axes[i].get_1dgrid()[self.local_slice(forward_output)[i]] for i in range(len(self.axes))]
        return self.xp.meshgrid(*grids, indexing='ij')

    def get_indices(self, forward_output=True):
        return [self.xp.arange(self.axes[i].N)[self.local_slice(forward_output)[i]] for i in range(len(self.axes))]

    @cache
    def get_pfft(self, axes=None, padding=None, grid=None):
        if self.ndim == 1 or self.comm is None:
            return None
        from mpi4py_fft import PFFT

        axes = tuple(i for i in range(self.ndim)) if axes is None else axes
        padding = list(padding if padding else [1.0 for _ in range(self.ndim)])

        def no_transform(u, *args, **kwargs):
            return u

        transforms = {(i,): (no_transform, no_transform) for i in range(self.ndim)}
        for i in axes:
            transforms[((i + self.ndim) % self.ndim,)] = (self.axes[i].transform, self.axes[i].itransform)

        # "transform" all axes to ensure consistent shapes.
        # Transform non-distributable axes last to ensure they are aligned
        _axes = tuple(sorted((axis + self.ndim) % self.ndim for axis in axes))
        _axes = [axis for axis in _axes if not self.axes[axis].distributable] + sorted(
            [axis for axis in _axes if self.axes[axis].distributable]
            + [axis for axis in range(self.ndim) if axis not in _axes]
        )

        pfft = PFFT(
            comm=self.comm,
            shape=self.global_shape[1:],
            axes=_axes,  # TODO: control the order of the transforms better
            dtype='D',
            collapse=False,
            backend=self.fft_backend,
            comm_backend=self.fft_comm_backend,
            padding=padding,
            transforms=transforms,
            grid=grid,
        )
        return pfft

    def get_fft(self, axes=None, direction='object', padding=None, shape=None):
        """
        When using MPI, we use `PFFT` objects generated by mpi4py-fft

        Args:
            axes (tuple): Axes you want to transform over
            direction (str): use "forward" or "backward" to get functions for performing the transforms or "object" to get the PFFT object
            padding (tuple): Padding for dealiasing
            shape (tuple): Shape of the transform

        Returns:
            transform
        """
        axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes
        shape = self.global_shape[1:] if shape is None else shape
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )
        key = (axes, direction, tuple(padding), tuple(shape))

        if key not in self.fft_cache.keys():
            if self.comm is None:
                assert np.allclose(padding, 1), 'Zero padding is not implemented for non-MPI transforms'

                if direction == 'forward':
                    self.fft_cache[key] = self.xp.fft.fftn
                elif direction == 'backward':
                    self.fft_cache[key] = self.xp.fft.ifftn
                elif direction == 'object':
                    self.fft_cache[key] = None
            else:
                if direction == 'object':
                    from mpi4py_fft import PFFT

                    _fft = PFFT(
                        comm=self.comm,
                        shape=shape,
                        axes=sorted(axes),
                        dtype='D',
                        collapse=False,
                        backend=self.fft_backend,
                        comm_backend=self.fft_comm_backend,
                        padding=padding,
                    )
                else:
                    _fft = self.get_fft(axes=axes, direction='object', padding=padding, shape=shape)

                if direction == 'forward':
                    self.fft_cache[key] = _fft.forward
                elif direction == 'backward':
                    self.fft_cache[key] = _fft.backward
                elif direction == 'object':
                    self.fft_cache[key] = _fft

        return self.fft_cache[key]

    def local_slice(self, forward_output=True):
        if self.fft_obj:
            return self.get_pfft().local_slice(forward_output=forward_output)
        else:
            return [slice(0, me.N) for me in self.axes]

    def global_slice(self, forward_output=True):
        if self.fft_obj:
            return [slice(0, me) for me in self.fft_obj.global_shape(forward_output=forward_output)]
        else:
            return self.local_slice(forward_output=forward_output)

    def setup_fft(self, real_spectral_coefficients=False):
        """
        This function must be called after all axes have been setup in order to prepare the local shapes of the data.
        This must also be called before setting up any BCs.

        Args:
            real_spectral_coefficients (bool): Allow only real coefficients in spectral space
        """
        if len(self.components) == 0:
            self.add_component('u')

        self.global_shape = (len(self.components),) + tuple(me.N for me in self.axes)

        axes = tuple(i for i in range(len(self.axes)))
        self.fft_obj = self.get_pfft(axes=axes)

        self.init = (
            np.empty(shape=self.global_shape)[
                (
                    ...,
                    *self.local_slice(False),
                )
            ].shape,
            self.comm,
            np.dtype('float'),
        )
        self.init_physical = (
            np.empty(shape=self.global_shape)[
                (
                    ...,
                    *self.local_slice(False),
                )
            ].shape,
            self.comm,
            np.dtype('float'),
        )
        self.init_forward = (
            np.empty(shape=self.global_shape)[
                (
                    ...,
                    *self.local_slice(True),
                )
            ].shape,
            self.comm,
            np.dtype('float') if real_spectral_coefficients else np.dtype('complex128'),
        )

        self.BC_mat = self.get_empty_operator_matrix()
        self.BC_rhs_mask = self.newDistArray().astype(bool)

    def newDistArray(self, pfft=None, forward_output=True, val=0, rank=1, view=False):
        """
        Get an empty distributed array. This is almost a copy of the function of the same name from mpi4py-fft, but
        takes care of all the solution components in the tensor.
        """
        if self.comm is None:
            return self.xp.zeros(self.init[0], dtype=self.init[2])
        from mpi4py_fft.distarray import DistArray

        pfft = pfft if pfft else self.get_pfft()
        if pfft is None:
            if forward_output:
                return self.u_init_forward
            else:
                return self.u_init

        global_shape = pfft.global_shape(forward_output)
        p0 = pfft.pencil[forward_output]
        if forward_output is True:
            dtype = pfft.forward.output_array.dtype
        else:
            dtype = pfft.forward.input_array.dtype
        global_shape = (self.ncomponents,) * rank + global_shape

        if pfft.xfftn[0].backend in ["cupy", "cupyx-scipy"]:
            from mpi4py_fft.distarrayCuPy import DistArrayCuPy as darraycls
        else:
            darraycls = DistArray

        z = darraycls(global_shape, subcomm=p0.subcomm, val=val, dtype=dtype, alignment=p0.axis, rank=rank)
        return z.v if view else z

    def infer_alignment(self, u, forward_output, padding=None, **kwargs):
        if self.comm is None:
            return [0]

        def _alignment(pfft):
            _arr = self.newDistArray(pfft, forward_output=forward_output)
            _aligned_axes = [i for i in range(self.ndim) if _arr.global_shape[i + 1] == u.shape[i + 1]]
            return _aligned_axes

        if padding is None:
            pfft = self.get_pfft(**kwargs)
            aligned_axes = _alignment(pfft)
        else:
            if self.ndim == 2:
                padding_options = [(1.0, padding[1]), (padding[0], 1.0), padding, (1.0, 1.0)]
            elif self.ndim == 3:
                padding_options = [
                    (1.0, 1.0, padding[2]),
                    (1.0, padding[1], 1.0),
                    (padding[0], 1.0, 1.0),
                    (1.0, padding[1], padding[2]),
                    (padding[0], 1.0, padding[2]),
                    (padding[0], padding[1], 1.0),
                    padding,
                    (1.0, 1.0, 1.0),
                ]
            else:
                raise NotImplementedError(f'Don\'t know how to infer alignment in {self.ndim}D!')
            for _padding in padding_options:
                pfft = self.get_pfft(padding=_padding, **kwargs)
                aligned_axes = _alignment(pfft)
                if len(aligned_axes) > 0:
                    self.logger.debug(
                        f'Found alignment of array with size {u.shape}: {aligned_axes} using padding {_padding}'
                    )
                    break

        assert len(aligned_axes) > 0, f'Found no aligned axes for array of size {u.shape}!'
        return aligned_axes

    def redistribute(self, u, axis, forward_output, **kwargs):
        if self.comm is None:
            return u

        pfft = self.get_pfft(**kwargs)
        _arr = self.newDistArray(pfft, forward_output=forward_output)

        if 'Dist' in type(u).__name__ and False:
            try:
                u.redistribute(out=_arr)
                return _arr
            except AssertionError:
                pass

        u_alignment = self.infer_alignment(u, forward_output=False, **kwargs)
        for alignment in u_alignment:
            _arr = _arr.redistribute(alignment)
            if _arr.shape == u.shape:
                _arr[...] = u
                return _arr.redistribute(axis)

        raise Exception(
            f'Don\'t know how to align array of local shape {u.shape} and global shape {self.global_shape}, aligned in axes {u_alignment}, to axis {axis}'
        )

    def transform(self, u, *args, axes=None, padding=None, **kwargs):
        pfft = self.get_pfft(*args, axes=axes, padding=padding, **kwargs)

        if pfft is None:
            axes = axes if axes else tuple(i for i in range(self.ndim))
            u_hat = u.copy()
            for i in axes:
                _axis = 1 + i if i >= 0 else i
                u_hat = self.axes[i].transform(u_hat, axes=(_axis,))
            return u_hat

        _in = self.newDistArray(pfft, forward_output=False, rank=1)
        _out = self.newDistArray(pfft, forward_output=True, rank=1)

        if _in.shape == u.shape:
            _in[...] = u
        else:
            _in[...] = self.redistribute(u, axis=_in.alignment, forward_output=False, padding=padding, **kwargs)

        for i in range(self.ncomponents):
            pfft.forward(_in[i], _out[i], normalize=False)

        if padding is not None:
            _out /= np.prod(padding)
        return _out

    def itransform(self, u, *args, axes=None, padding=None, **kwargs):
        if padding is not None:
            assert all(
                (self.axes[i].N * padding[i]) % 1 == 0 for i in range(self.ndim)
            ), 'Cannot do this padding with this resolution. Resulting resolution must be integer'

        pfft = self.get_pfft(*args, axes=axes, padding=padding, **kwargs)
        if pfft is None:
            axes = axes if axes else tuple(i for i in range(self.ndim))
            u_hat = u.copy()
            for i in axes:
                _axis = 1 + i if i >= 0 else i
                u_hat = self.axes[i].itransform(u_hat, axes=(_axis,))
            return u_hat

        _in = self.newDistArray(pfft, forward_output=True, rank=1)
        _out = self.newDistArray(pfft, forward_output=False, rank=1)

        if _in.shape == u.shape:
            _in[...] = u
        else:
            _in[...] = self.redistribute(u, axis=_in.alignment, forward_output=True, padding=padding, **kwargs)

        for i in range(self.ncomponents):
            pfft.backward(_in[i], _out[i], normalize=True)

        if padding is not None:
            _out *= np.prod(padding)
        return _out

    def get_local_slice_of_1D_matrix(self, M, axis):
        """
        Get the local version of a 1D matrix. When using distributed FFTs, each rank will carry only a subset of modes,
        which you can sort out via the `SpectralHelper.local_slice()` attribute. When constructing a 1D matrix, you can
        use this method to get the part corresponding to the modes carried by this rank.

        Args:
            M (sparse matrix): Global 1D matrix you want to get the local version of
            axis (int): Direction in which you want the local version. You will get the global matrix in other directions.

        Returns:
            sparse local matrix
        """
        return M.tocsc()[self.local_slice(True)[axis], self.local_slice(True)[axis]]

    def expand_matrix_ND(self, matrix, aligned):
        sp = self.sparse_lib
        axes = np.delete(np.arange(self.ndim), aligned)
        ndim = len(axes) + 1

        if ndim == 1:
            mat = matrix
        elif ndim == 2:
            axis = axes[0]
            I1D = sp.eye(self.axes[axis].N)

            mats = [None] * ndim
            mats[aligned] = self.get_local_slice_of_1D_matrix(matrix, aligned)
            mats[axis] = self.get_local_slice_of_1D_matrix(I1D, axis)

            mat = sp.kron(*mats)
        elif ndim == 3:

            mats = [None] * ndim
            mats[aligned] = self.get_local_slice_of_1D_matrix(matrix, aligned)
            for axis in axes:
                I1D = sp.eye(self.axes[axis].N)
                mats[axis] = self.get_local_slice_of_1D_matrix(I1D, axis)

            mat = sp.kron(mats[0], sp.kron(*mats[1:]))

        else:
            raise NotImplementedError(f'Matrix expansion not implemented for {ndim} dimensions!')

        mat = self.eliminate_zeros(mat)
        return mat

    def get_filter_matrix(self, axis, **kwargs):
        """
        Get bandpass filter along `axis`. See the documentation `get_filter_matrix` in the 1D bases for what kwargs are
        admissible.

        Returns:
            sparse bandpass matrix
        """
        if self.ndim == 1:
            return self.axes[0].get_filter_matrix(**kwargs)

        mats = [base.get_Id() for base in self.axes]
        mats[axis] = self.axes[axis].get_filter_matrix(**kwargs)
        return self.sparse_lib.kron(*mats)

    def get_differentiation_matrix(self, axes, **kwargs):
        """
        Get differentiation matrix along specified axis. `kwargs` are forwarded to the 1D base implementation.

        Args:
            axes (tuple): Axes along which to differentiate.

        Returns:
            sparse differentiation matrix
        """
        D = self.expand_matrix_ND(self.axes[axes[0]].get_differentiation_matrix(**kwargs), axes[0])
        for axis in axes[1:]:
            _D = self.axes[axis].get_differentiation_matrix(**kwargs)
            D = D @ self.expand_matrix_ND(_D, axis)

        return D

    def get_integration_matrix(self, axes):
        """
        Get integration matrix to integrate along specified axis.

        Args:
            axes (tuple): Axes along which to integrate over.

        Returns:
            sparse integration matrix
        """
        S = self.expand_matrix_ND(self.axes[axes[0]].get_integration_matrix(), axes[0])
        for axis in axes[1:]:
            _S = self.axes[axis].get_integration_matrix()
            S = S @ self.expand_matrix_ND(_S, axis)

        return S

    def get_Id(self):
        """
        Get identity matrix

        Returns:
            sparse identity matrix
        """
        I = self.expand_matrix_ND(self.axes[0].get_Id(), 0)
        for axis in range(1, self.ndim):
            _I = self.axes[axis].get_Id()
            I = I @ self.expand_matrix_ND(_I, axis)
        return I

    def get_Dirichlet_recombination_matrix(self, axis=-1):
        """
        Get Dirichlet recombination matrix along axis. Not that it only makes sense in directions discretized with variations of Chebychev bases.

        Args:
            axis (int): Axis you discretized with Chebychev

        Returns:
            sparse matrix
        """
        C1D = self.axes[axis].get_Dirichlet_recombination_matrix()
        return self.expand_matrix_ND(C1D, axis)

    def get_basis_change_matrix(self, axes=None, **kwargs):
        """
        Some spectral bases do a change between bases while differentiating. This method returns matrices that changes the basis to whatever you want.
        Refer to the methods of the same name of the 1D bases to learn what parameters you need to pass here as `kwargs`.

        Args:
            axes (tuple): Axes along which to change basis.

        Returns:
            sparse basis change matrix
        """
        axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes

        C = self.expand_matrix_ND(self.axes[axes[0]].get_basis_change_matrix(**kwargs), axes[0])
        for axis in axes[1:]:
            _C = self.axes[axis].get_basis_change_matrix(**kwargs)
            C = C @ self.expand_matrix_ND(_C, axis)

        return C
