import numpy as np
import scipy
from pySDC.implementations.datatype_classes.mesh import mesh
from scipy.special import factorial


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

    def __init__(self, N, x0=None, x1=None, useGPU=False):
        """
        Constructor

        Args:
            N (int): Resolution
            x0 (float): Coordinate of left boundary
            x1 (float): Coordinate of right boundary
            useGPU (bool): Whether to use GPUs
        """
        self.N = N
        self.x0 = x0
        self.x1 = x1
        self.L = x1 - x0
        self.useGPU = useGPU

        if useGPU:
            self.setup_GPU()

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

    This implementation is largely based on the Dedalus paper (arXiv:1905.10388).
    """

    def __init__(self, *args, transform_type='fft', x0=-1, x1=1, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            transform_type ('fft' or 'dct'): Either use DCT functions directly implemented in the transform library or
                                             use the FFT from the library to compute the DCT
            x0 (float): Coordinate of left boundary. Note that only -1 is currently implented
            x1 (float): Coordinate of right boundary. Note that only +1 is currently implented
        """
        # need linear transformation y = ax + b with a = (x1-x0)/2 and b = (x1+x0)/2
        self.lin_trf_fac = (x1 - x0) / 2
        self.lin_trf_off = (x1 + x0) / 2
        super().__init__(*args, x0=x0, x1=x1, **kwargs)
        self.transform_type = transform_type

        if self.transform_type == 'fft':
            self.get_fft_utils()

        self.cache = {}
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
        if name in self.cache.keys() and not N:
            return self.cache[name]

        N = N if N else self.N
        sp = self.sparse_lib
        xp = self.xp

        def get_forward_conv(name):
            if name == 'T2U':
                mat = (sp.eye(N) - sp.diags(xp.ones(N - 2), offsets=+2)) / 2.0
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

        self.cache[name] = mat
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

    def get_fft_shuffle(self, forward, N):
        """
        In order to more easily parallelize using distributed FFT libraries, we express the DCT via an FFT following
        doi.org/10.1109/TASSP.1980.1163351. The idea is based on reshuffling the data to be periodic and rotating it
        in the complex plane. This function returns a mask to do the shuffling.

        Args:
            forward (bool): Whether you want the shuffle for forward transform or backward transform
            N (int): size of the grid

        Returns:
            self.xp.array: Use as mask
        """
        xp = self.xp
        if forward:
            return xp.append(xp.arange((N + 1) // 2) * 2, -xp.arange(N // 2) * 2 - 1 - N % 2)
        else:
            mask = xp.zeros(N, dtype=int)
            mask[: N - N % 2 : 2] = xp.arange(N // 2)
            mask[1::2] = N - xp.arange(N // 2) - 1
            mask[-1] = N // 2
            return mask

    def get_fft_shift(self, forward, N):
        """
        As described in the docstring for `get_fft_shuffle`, we need to rotate in the complex plane in order to use FFT for DCT.

        Args:
            forward (bool): Whether you want the rotation for forward transform or backward transform
            N (int): size of the grid

        Returns:
            self.xp.array: Rotation
        """
        k = self.get_wavenumbers()
        norm = self.get_norm()
        xp = self.xp
        if forward:
            return 2 * xp.exp(-1j * np.pi * k / (2 * N) + 0j * np.pi / 4) * norm
        else:
            shift = xp.exp(1j * np.pi * k / (2 * N))
            shift[0] = 0.5
            return shift / norm

    def get_fft_utils(self):
        """
        Get the required utilities for using FFT to do DCT as described in the docstring for `get_fft_shuffle` and keep
        them cached.
        """
        self.fft_utils = {
            'fwd': {},
            'bck': {},
        }

        # forwards transform
        self.fft_utils['fwd']['shuffle'] = self.get_fft_shuffle(True, self.N)
        self.fft_utils['fwd']['shift'] = self.get_fft_shift(True, self.N)

        # backwards transform
        self.fft_utils['bck']['shuffle'] = self.get_fft_shuffle(False, self.N)
        self.fft_utils['bck']['shift'] = self.get_fft_shift(False, self.N)

        return self.fft_utils

    def transform(self, u, axis=-1, **kwargs):
        """
        1D DCT along axis. `kwargs` will be passed on to the FFT library.

        Args:
            u: Data you want to transform
            axis (int): Axis you want to transform along

        Returns:
            Data in spectral space
        """
        if self.transform_type.lower() == 'dct':
            return self.fft_lib.dct(u, axis=axis, **kwargs) * self.norm
        elif self.transform_type.lower() == 'fft':
            result = u.copy()

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.fft_utils['fwd']['shuffle']

            v = u[(*shuffle,)]

            V = self.fft_lib.fft(v, axis=axis, **kwargs)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            V *= self.fft_utils['fwd']['shift'][(*expansion,)]

            result.real[...] = V.real[...]
            return result
        else:
            raise NotImplementedError(f'Please choose a transform type from fft and dct, not {self.transform_type=}')

    def itransform(self, u, axis=-1):
        """
        1D inverse DCT along axis.

        Args:
            u: Data you want to transform
            axis (int): Axis you want to transform along

        Returns:
            Data in physical space
        """
        assert self.norm.shape[0] == u.shape[axis]

        if self.transform_type == 'dct':
            return self.fft_lib.idct(u / self.norm, axis=axis)
        elif self.transform_type == 'fft':
            result = u.copy()

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            v = self.fft_lib.ifft(u * self.fft_utils['bck']['shift'][(*expansion,)], axis=axis)

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.fft_utils['bck']['shuffle']
            V = v[(*shuffle,)]

            result.real[...] = V.real[...]
            return result
        else:
            raise NotImplementedError

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
    def __init__(self, *args, x0=0, x1=2 * np.pi, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            transform_type ('fft' or 'dct'): Either use DCT functions directly implemented in the transform library or
                                             use the FFT from the library to compute the DCT
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
            # Have to raise the matrix to power p on CPU because the GPU equivalent is not implemented in CuPy at the time of writing.
            import scipy.sparse as sp

            D = self.sparse_lib.diags(1j * k).get()
            return self.sparse_lib.csc_matrix(sp.linalg.matrix_power(D, p))
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

    def transform(self, u, axis=-1, **kwargs):
        """
        1D FFT along axis. `kwargs` are passed on to the FFT library.

        Args:
            u: Data you want to transform
            axis (int): Axis you want to transform along

        Returns:
            transformed data
        """
        return self.fft_lib.fft(u, axis=axis, **kwargs)

    def itransform(self, u, axis=-1):
        """
        Inverse 1D FFT.

        Args:
            u: Data you want to transform
            axis (int): Axis you want to transform along

        Returns:
            transformed data
        """
        return self.fft_lib.ifft(u, axis=axis)

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
        local_slice (slice): Local slice of the solution as in `mpi4py-fft`
        fft_obj: When using distributed FFTs, this will be a parallel transform object from `mpi4py-fft`
        init (tuple): This is the same `init` that is used throughout the problem classes
        init_forward (tuple): This is the equivalent of `init` in spectral space
    """

    xp = np
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    linalg = scipy.sparse.linalg
    dtype = mesh
    fft_backend = 'fftw'
    fft_comm_backend = 'MPI'

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        import cupyx.scipy.sparse as sparse_lib
        import cupyx.scipy.sparse.linalg as linalg
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

        cls.xp = cp
        cls.sparse_lib = sparse_lib
        cls.linalg = linalg

        cls.fft_backend = 'cupy'
        cls.fft_comm_backend = 'NCCL'

        cls.dtype = cupy_mesh

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

        self.axes = []
        self.components = []

        self.full_BCs = []
        self.BC_mat = None
        self.BCs = None

        self.fft_cache = {}
        self.fft_dealias_shape_cache = {}

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
            kwargs['transform_type'] = kwargs.get('transform_type', 'fft')
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

    def get_empty_operator_matrix(self):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Returns:
            list containing sparse zeros
        """
        S = len(self.components)
        O = self.get_Id() * 0
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
            return self.sparse_lib.csc_matrix(BC)
        elif ndim == 2:
            axis2 = (axis + 1) % ndim

            if scalar:
                _Id = self.sparse_lib.diags(self.xp.append([1], self.xp.zeros(self.axes[axis2].N - 1)))
            else:
                _Id = self.axes[axis2].get_Id()

            Id = self.get_local_slice_of_1D_matrix(self.axes[axis2].get_Id() @ _Id, axis=axis2)

            if self.useGPU:
                Id = Id.get()

            mats = [
                None,
            ] * ndim
            mats[axis] = self.get_local_slice_of_1D_matrix(BC, axis=axis)
            mats[axis2] = Id
            return self.sparse_lib.csc_matrix(sp.kron(*mats))

    def remove_BC(self, component, equation, axis, kind, line=-1, scalar=False, **kwargs):
        """
        Remove a BC from the matrix. This is useful e.g. when you add a non-scalar BC and then need to selectively
        remove single BCs again, as in incompressible Navier-Stokes, for instance.
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
        if (N + line) % N in self.xp.arange(N)[self.local_slice[axis]]:
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
            slices = (
                [self.index(equation)]
                + [slice(0, self.init[0][i + 1]) for i in range(axis)]
                + [line]
                + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
            )
            N = self.axes[axis].N
            if (N + line) % N in self.xp.arange(N)[self.local_slice[axis]]:
                slices[axis + 1] -= self.local_slice[axis].start
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
            self._rhs_hat_zero_mask = self.xp.zeros(shape=rhs_hat.shape, dtype=bool)

            for axis in range(self.ndim):
                for bc in self.full_BCs:
                    slices = (
                        [slice(0, self.init[0][i + 1]) for i in range(axis)]
                        + [bc['line']]
                        + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
                    )
                    if axis == bc['axis']:
                        _slice = [self.index(bc['equation'])] + slices
                        N = self.axes[axis].N
                        if (N + bc['line']) % N in self.xp.arange(N)[self.local_slice[axis]]:
                            _slice[axis + 1] -= self.local_slice[axis].start
                            self._rhs_hat_zero_mask[(*_slice,)] = True

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
                slices = (
                    [slice(0, self.init[0][i + 1]) for i in range(axis)]
                    + [bc['line']]
                    + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
                )
                if axis == bc['axis']:
                    _slice = [self.index(bc['equation'])] + slices

                    N = self.axes[axis].N
                    if (N + bc['line']) % N in self.xp.arange(N)[self.local_slice[axis]]:
                        _slice[axis + 1] -= self.local_slice[axis].start

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
            return M[0][0]
        else:
            return self.sparse_lib.bmat(M, format='csc')

    def get_wavenumbers(self):
        """
        Get grid in spectral space
        """
        grids = [self.axes[i].get_wavenumbers()[self.local_slice[i]] for i in range(len(self.axes))][::-1]
        return self.xp.meshgrid(*grids)

    def get_grid(self):
        """
        Get grid in physical space
        """
        grids = [self.axes[i].get_1dgrid()[self.local_slice[i]] for i in range(len(self.axes))][::-1]
        return self.xp.meshgrid(*grids)

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
        self.local_slice = [slice(0, me.N) for me in self.axes]

        axes = tuple(i for i in range(len(self.axes)))
        self.fft_obj = self.get_fft(axes=axes, direction='object')
        if self.fft_obj is not None:
            self.local_slice = self.fft_obj.local_slice(False)

        self.init = (
            np.empty(shape=self.global_shape)[
                (
                    ...,
                    *self.local_slice,
                )
            ].shape,
            self.comm,
            np.dtype('float'),
        )
        self.init_forward = (
            np.empty(shape=self.global_shape)[
                (
                    ...,
                    *self.local_slice,
                )
            ].shape,
            self.comm,
            np.dtype('float') if real_spectral_coefficients else np.dtype('complex128'),
        )

        self.BC_mat = self.get_empty_operator_matrix()
        self.BC_rhs_mask = self.xp.zeros(
            shape=self.init[0],
            dtype=bool,
        )

    def _transform_fft(self, u, axes, **kwargs):
        """
        FFT along `axes`

        Args:
            u: The solution
            axes (tuple): Axes you want to transform over

        Returns:
            transformed solution
        """
        # TODO: clean up and try putting more of this in the 1D bases
        fft = self.get_fft(axes, 'forward', **kwargs)
        return fft(u, axes=axes)

    def _transform_dct(self, u, axes, padding=None, **kwargs):
        '''
        DCT along `axes`.
        This will only return real values!
        When padding the solution, we cannot just use the mpi4py-fft implementation, because of the unusual ordering of
        wavenumbers in FFTs.

        Args:
            u: The solution
            axes (tuple): Axes you want to transform over

        Returns:
            transformed solution
        '''
        # TODO: clean up and try putting more of this in the 1D bases
        if self.debug:
            assert self.xp.allclose(u.imag, 0), 'This function can only handle real input.'

        if len(axes) > 1:
            v = self._transform_dct(self._transform_dct(u, axes[1:], **kwargs), (axes[0],), **kwargs)
        else:
            v = u.copy().astype(complex)
            axis = axes[0]
            base = self.axes[axis]

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = base.get_fft_shuffle(True, N=v.shape[axis])
            v = v[(*shuffle,)]

            if padding is not None:
                shape = list(v.shape)
                if ('forward', *padding) in self.fft_dealias_shape_cache.keys():
                    shape[0] = self.fft_dealias_shape_cache[('forward', *padding)]
                elif self.comm:
                    send_buf = np.array(v.shape[0])
                    recv_buf = np.array(v.shape[0])
                    self.comm.Allreduce(send_buf, recv_buf)
                    shape[0] = int(recv_buf)
                fft = self.get_fft(axes, 'forward', shape=shape)
            else:
                fft = self.get_fft(axes, 'forward', **kwargs)

            v = fft(v, axes=axes)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, v.shape[axis], 1)

            if padding is not None:
                shift = base.get_fft_shift(True, v.shape[axis])

                if padding[axis] != 1:
                    N = int(np.ceil(v.shape[axis] / padding[axis]))
                    _expansion = [slice(0, n) for n in v.shape]
                    _expansion[axis] = slice(0, N, 1)
                    v = v[(*_expansion,)]
            else:
                shift = base.fft_utils['fwd']['shift']

            v *= shift[(*expansion,)]

        return v.real

    def transform_single_component(self, u, axes=None, padding=None):
        """
        Transform a single component of the solution

        Args:
            u data to transform:
            axes (tuple): Axes over which to transform
            padding (list): Padding factor for transform. E.g. a padding factor of 2 will discard the upper half of modes after transforming

        Returns:
            Transformed data
        """
        # TODO: clean up and try putting more of this in the 1D bases
        trfs = {
            ChebychevHelper: self._transform_dct,
            UltrasphericalHelper: self._transform_dct,
            FFTHelper: self._transform_fft,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )  # You know, sometimes I feel very strongly about Black still. This atrocious formatting is readable by Sauron only.

        result = u.copy().astype(complex)
        alignment = self.ndim - 1

        axes_collapsed = [tuple(sorted(me for me in axes if type(self.axes[me]) == base)) for base in trfs.keys()]
        bases = [list(trfs.keys())[i] for i in range(len(axes_collapsed)) if len(axes_collapsed[i]) > 0]
        axes_collapsed = [me for me in axes_collapsed if len(me) > 0]
        shape = [max(u.shape[i], self.global_shape[1 + i]) for i in range(self.ndim)]

        fft = self.get_fft(axes=axes, padding=padding, direction='object')
        if fft is not None:
            shape = list(fft.global_shape(False))

        for trf in range(len(axes_collapsed)):
            _axes = axes_collapsed[trf]
            base = bases[trf]

            if len(_axes) == 0:
                continue

            for _ax in _axes:
                shape[_ax] = self.global_shape[1 + self.ndim + _ax]

            fft = self.get_fft(_axes, 'object', padding=padding, shape=shape)

            _in = self.get_aligned(
                result, axis_in=alignment, axis_out=self.ndim + _axes[-1], forward=False, fft=fft, shape=shape
            )

            alignment = self.ndim + _axes[-1]

            _out = trfs[base](_in, axes=_axes, padding=padding, shape=shape)

            if self.comm is not None:
                _out *= np.prod([self.axes[i].N for i in _axes])

            axes_next_base = (axes_collapsed + [(-1,)])[trf + 1]
            alignment = alignment if len(axes_next_base) == 0 else self.ndim + axes_next_base[-1]
            result = self.get_aligned(
                _out, axis_in=self.ndim + _axes[0], axis_out=alignment, fft=fft, forward=True, shape=shape
            )

        return result

    def transform(self, u, axes=None, padding=None):
        """
        Transform all components from physical space to spectral space

        Args:
            u data to transform:
            axes (tuple): Axes over which to transform
            padding (list): Padding factor for transform. E.g. a padding factor of 2 will discard the upper half of modes after transforming

        Returns:
            Transformed data
        """
        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        result = [
            None,
        ] * self.ncomponents
        for comp in self.components:
            i = self.index(comp)

            result[i] = self.transform_single_component(u[i], axes=axes, padding=padding)

        return self.xp.stack(result)

    def _transform_ifft(self, u, axes, **kwargs):
        # TODO: clean up and try putting more of this in the 1D bases
        ifft = self.get_fft(axes, 'backward', **kwargs)
        return ifft(u, axes=axes)

    def _transform_idct(self, u, axes, padding=None, **kwargs):
        '''
        This will only ever return real values!
        '''
        # TODO: clean up and try putting more of this in the 1D bases
        if self.debug:
            assert self.xp.allclose(u.imag, 0), 'This function can only handle real input.'

        v = u.copy().astype(complex)

        if len(axes) > 1:
            v = self._transform_idct(self._transform_idct(u, axes[1:]), (axes[0],))
        else:
            axis = axes[0]
            base = self.axes[axis]

            if padding is not None:
                if padding[axis] != 1:
                    N_pad = int(np.ceil(v.shape[axis] * padding[axis]))
                    _pad = [[0, 0] for _ in v.shape]
                    _pad[axis] = [0, N_pad - base.N]
                    v = self.xp.pad(v, _pad, 'constant')

                    shift = self.xp.exp(1j * np.pi * self.xp.arange(N_pad) / (2 * N_pad)) * base.N
                else:
                    shift = base.fft_utils['bck']['shift']
            else:
                shift = base.fft_utils['bck']['shift']

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, v.shape[axis], 1)

            v *= shift[(*expansion,)]

            if padding is not None:
                if padding[axis] != 1:
                    shape = list(v.shape)
                    if ('backward', *padding) in self.fft_dealias_shape_cache.keys():
                        shape[0] = self.fft_dealias_shape_cache[('backward', *padding)]
                    elif self.comm:
                        send_buf = np.array(v.shape[0])
                        recv_buf = np.array(v.shape[0])
                        self.comm.Allreduce(send_buf, recv_buf)
                        shape[0] = int(recv_buf)
                    ifft = self.get_fft(axes, 'backward', shape=shape)
                else:
                    ifft = self.get_fft(axes, 'backward', padding=padding, **kwargs)
            else:
                ifft = self.get_fft(axes, 'backward', padding=padding, **kwargs)
            v = ifft(v, axes=axes)

            shuffle = [slice(0, s, 1) for s in v.shape]
            shuffle[axis] = base.get_fft_shuffle(False, N=v.shape[axis])
            v = v[(*shuffle,)]

        return v.real

    def itransform_single_component(self, u, axes=None, padding=None):
        """
        Inverse transform over single component of the solution

        Args:
            u data to transform:
            axes (tuple): Axes over which to transform
            padding (list): Padding factor for transform. E.g. a padding factor of 2 will add as many zeros as there were modes before before transforming

        Returns:
            Transformed data
        """
        # TODO: clean up and try putting more of this in the 1D bases
        trfs = {
            FFTHelper: self._transform_ifft,
            ChebychevHelper: self._transform_idct,
            UltrasphericalHelper: self._transform_idct,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        result = u.copy().astype(complex)
        alignment = self.ndim - 1

        axes_collapsed = [tuple(sorted(me for me in axes if type(self.axes[me]) == base)) for base in trfs.keys()]
        bases = [list(trfs.keys())[i] for i in range(len(axes_collapsed)) if len(axes_collapsed[i]) > 0]
        axes_collapsed = [me for me in axes_collapsed if len(me) > 0]
        shape = list(self.global_shape[1:])

        for trf in range(len(axes_collapsed)):
            _axes = axes_collapsed[trf]
            base = bases[trf]

            if len(_axes) == 0:
                continue

            fft = self.get_fft(_axes, 'object', padding=padding, shape=shape)

            _in = self.get_aligned(
                result, axis_in=alignment, axis_out=self.ndim + _axes[0], forward=True, fft=fft, shape=shape
            )
            if self.comm is not None:
                _in /= np.prod([self.axes[i].N for i in _axes])

            alignment = self.ndim + _axes[0]

            _out = trfs[base](_in, axes=_axes, padding=padding, shape=shape)

            for _ax in _axes:
                if fft:
                    shape[_ax] = fft._input_shape[_ax]
                else:
                    shape[_ax] = _out.shape[_ax]

            axes_next_base = (axes_collapsed + [(-1,)])[trf + 1]
            alignment = alignment if len(axes_next_base) == 0 else self.ndim + axes_next_base[0]
            result = self.get_aligned(
                _out, axis_in=self.ndim + _axes[-1], axis_out=alignment, fft=fft, forward=False, shape=shape
            )

        return result

    def get_aligned(self, u, axis_in, axis_out, fft=None, forward=False, **kwargs):
        """
        Realign the data along the axis when using distributed FFTs. `kwargs` will be used to get the correct PFFT
        object from `mpi4py-fft`, which has suitable transfer classes for the shape of data. Hence, they should include
        shape especially, if applicable.

        Args:
            u: The solution
            axis_in (int): Current alignment
            axis_out (int): New alignment
            fft (mpi4py_fft.PFFT), optional: parallel FFT object
            forward (bool): Whether the input is in spectral space or not

        Returns:
            solution aligned on `axis_in`
        """
        if self.comm is None or axis_in == axis_out:
            return u.copy()
        if self.comm.size == 1:
            return u.copy()

        global_fft = self.get_fft(**kwargs)
        axisA = [me.axisA for me in global_fft.transfer]
        axisB = [me.axisB for me in global_fft.transfer]

        current_axis = axis_in

        if axis_in in axisA and axis_out in axisB:
            while current_axis != axis_out:
                transfer = global_fft.transfer[axisA.index(current_axis)]

                arrayB = self.xp.empty(shape=transfer.subshapeB, dtype=transfer.dtype)
                arrayA = self.xp.empty(shape=transfer.subshapeA, dtype=transfer.dtype)
                arrayA[:] = u[:]

                transfer.forward(arrayA=arrayA, arrayB=arrayB)

                current_axis = transfer.axisB
                u = arrayB

            return u
        elif axis_in in axisB and axis_out in axisA:
            while current_axis != axis_out:
                transfer = global_fft.transfer[axisB.index(current_axis)]

                arrayB = self.xp.empty(shape=transfer.subshapeB, dtype=transfer.dtype)
                arrayA = self.xp.empty(shape=transfer.subshapeA, dtype=transfer.dtype)
                arrayB[:] = u[:]

                transfer.backward(arrayA=arrayA, arrayB=arrayB)

                current_axis = transfer.axisA
                u = arrayA

            return u
        else:  # go the potentially slower route of not reusing transfer classes
            from mpi4py_fft import newDistArray

            fft = self.get_fft(**kwargs) if fft is None else fft

            _in = newDistArray(fft, forward).redistribute(axis_in)
            _in[...] = u

            return _in.redistribute(axis_out)

    def itransform(self, u, axes=None, padding=None):
        """
        Inverse transform over all components of the solution

        Args:
            u data to transform:
            axes (tuple): Axes over which to transform
            padding (list): Padding factor for transform. E.g. a padding factor of 2 will add as many zeros as there were modes before before transforming

        Returns:
            Transformed data
        """
        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        result = [
            None,
        ] * self.ncomponents
        for comp in self.components:
            i = self.index(comp)

            result[i] = self.itransform_single_component(u[i], axes=axes, padding=padding)

        return self.xp.stack(result)

    def get_local_slice_of_1D_matrix(self, M, axis):
        """
        Get the local version of a 1D matrix. When using distributed FFTs, each rank will carry only a subset of modes,
        which you can sort out via the `SpectralHelper.local_slice` attribute. When constructing a 1D matrix, you can
        use this method to get the part corresponding to the modes carried by this rank.

        Args:
            M (sparse matrix): Global 1D matrix you want to get the local version of
            axis (int): Direction in which you want the local version. You will get the global matrix in other directions. This means slab decomposition only.

        Returns:
            sparse local matrix
        """
        return M.tocsc()[self.local_slice[axis], self.local_slice[axis]]

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
        sp = self.sparse_lib
        ndim = self.ndim

        if ndim == 1:
            D = self.axes[0].get_differentiation_matrix(**kwargs)
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                D1D = self.axes[axis].get_differentiation_matrix(**kwargs)

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(D1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    D = sp.kron(*mats)
                else:
                    D = D @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Differentiation matrix not implemented for {ndim} dimension!')

        return D

    def get_integration_matrix(self, axes):
        """
        Get integration matrix to integrate along specified axis.

        Args:
            axes (tuple): Axes along which to integrate over.

        Returns:
            sparse integration matrix
        """
        sp = self.sparse_lib
        ndim = len(self.axes)

        if ndim == 1:
            S = self.axes[0].get_integration_matrix()
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                S1D = self.axes[axis].get_integration_matrix()

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(S1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    S = sp.kron(*mats)
                else:
                    S = S @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Integration matrix not implemented for {ndim} dimension!')

        return S

    def get_Id(self):
        """
        Get identity matrix

        Returns:
            sparse identity matrix
        """
        sp = self.sparse_lib
        ndim = self.ndim
        I = sp.eye(np.prod(self.init[0][1:]), dtype=complex)

        if ndim == 1:
            I = self.axes[0].get_Id()
        elif ndim == 2:
            for axis in range(ndim):
                axis2 = (axis + 1) % ndim
                I1D = self.axes[axis].get_Id()

                I1D2 = sp.eye(self.axes[axis2].N)

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(I1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D2, axis2)

                I = I @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Identity matrix not implemented for {ndim} dimension!')

        return I

    def get_Dirichlet_recombination_matrix(self, axis=-1):
        """
        Get Dirichlet recombination matrix along axis. Not that it only makes sense in directions discretized with variations of Chebychev bases.

        Args:
            axis (int): Axis you discretized with Chebychev

        Returns:
            sparse matrix
        """
        sp = self.sparse_lib
        ndim = len(self.axes)

        if ndim == 1:
            C = self.axes[0].get_Dirichlet_recombination_matrix()
        elif ndim == 2:
            axis2 = (axis + 1) % ndim
            C1D = self.axes[axis].get_Dirichlet_recombination_matrix()

            I1D = self.axes[axis2].get_Id()

            mats = [None] * ndim
            mats[axis] = self.get_local_slice_of_1D_matrix(C1D, axis)
            mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

            C = sp.kron(*mats)
        else:
            raise NotImplementedError(f'Basis change matrix not implemented for {ndim} dimension!')

        return C

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

        sp = self.sparse_lib
        ndim = len(self.axes)

        if ndim == 1:
            C = self.axes[0].get_basis_change_matrix(**kwargs)
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                C1D = self.axes[axis].get_basis_change_matrix(**kwargs)

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(C1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    C = sp.kron(*mats)
                else:
                    C = C @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Basis change matrix not implemented for {ndim} dimension!')

        return C
