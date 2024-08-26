import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT
from mpi4py_fft import newDistArray


class allencahn_imex(IMEX_Laplacian_MPIFFT):
    r"""
    Example implementing the :math:`2`-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [0, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`, driving force :math:`d_w`, and :math:`N=2,3`. Different initial
    conditions can be used, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{(x_i-0.5)^2 + (y_j-0.5)^2}}{\sqrt{2}\varepsilon}\right),

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is treated
    *semi-implicitly*, i.e., the linear part is solved with Fast-Fourier Transform (FFT) and the nonlinear part in the right-hand
    side will be treated explicitly using ``mpi4py-fft`` [1]_ to solve them.

    Parameters
    ----------
    nvars : List of int tuples, optional
        Number of unknowns in the problem, e.g. ``nvars=(128, 128)``.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    radius : float, optional
        Radius of the circles.
    spectral : bool, optional
        Indicates if spectral initial condition is used.
    dw : float, optional
        Driving force.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str, optional
        Initialises type of initial state.
    comm : bool, optional
        Communicator for parallelization.

    Attributes
    ----------
    fft : fft object
        Object for FFT.
    X : np.ogrid
        Grid coordinates in real space.
    K2 : np.1darray
        Laplace operator in spectral space.
    dx : float
        Mesh width in x direction.
    dy : float
        Mesh width in y direction.

    References
    ----------
    .. [1] https://mpi4py-fft.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        eps=0.04,
        radius=0.25,
        dw=0.0,
        init_type='circle',
        **kwargs,
    ):
        kwargs['L'] = kwargs.get('L', 1.0)
        super().__init__(alpha=1.0, dtype=np.dtype('float'), **kwargs)
        self._makeAttributeAndRegister('eps', 'radius', 'dw', 'init_type', localVars=locals(), readOnly=True)

    def _eval_explicit_part(self, u, t, f_expl):
        f_expl[:] = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * self.dw * u * (1.0 - u)
        return f_expl

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        f.impl[:] = self._eval_Laplacian(u, f.impl)

        if self.spectral:
            f.impl = -self.K2 * u

            if self.eps > 0:
                tmp = self.fft.backward(u)
                tmp[:] = self._eval_explicit_part(tmp, t, tmp)
                f.expl[:] = self.fft.forward(tmp)

        else:

            if self.eps > 0:
                f.expl[:] = self._eval_explicit_part(u, t, f.expl)

        self.work_counters['rhs']()
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.init_type == 'circle':
            r2 = (self.X[0] - 0.5) ** 2 + (self.X[1] - 0.5) ** 2
            if self.spectral:
                tmp = 0.5 * (1.0 + self.xp.tanh((self.radius - self.xp.sqrt(r2)) / (np.sqrt(2) * self.eps)))
                me[:] = self.fft.forward(tmp)
            else:
                me[:] = 0.5 * (1.0 + self.xp.tanh((self.radius - self.xp.sqrt(r2)) / (np.sqrt(2) * self.eps)))
        elif self.init_type == 'circle_rand':
            ndim = len(me.shape)
            L = int(self.L[0])
            # get random radii for circles/spheres
            self.xp.random.seed(1)
            lbound = 3.0 * self.eps
            ubound = 0.5 - self.eps
            rand_radii = (ubound - lbound) * self.xp.random.random_sample(size=tuple([L] * ndim)) + lbound
            # distribute circles/spheres
            tmp = newDistArray(self.fft, False)
            if ndim == 2:
                for i in range(0, L):
                    for j in range(0, L):
                        # build radius
                        r2 = (self.X[0] + i - L + 0.5) ** 2 + (self.X[1] + j - L + 0.5) ** 2
                        # add this blob, shifted by 1 to avoid issues with adding up negative contributions
                        tmp += self.xp.tanh((rand_radii[i, j] - self.xp.sqrt(r2)) / (np.sqrt(2) * self.eps)) + 1
            else:
                raise NotImplementedError
            # normalize to [0,1]
            tmp *= 0.5
            assert self.xp.all(tmp <= 1.0)
            if self.spectral:
                me[:] = self.fft.forward(tmp)
            else:
                self.xp.copyto(me, tmp)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.init_type)

        return me


class allencahn_imex_timeforcing(allencahn_imex):
    r"""
    Example implementing the :math:`N`-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [0, 1]^2`
    using time-dependent forcing

    .. math::
        \frac{\partial u}{\partial t} = \Delta u - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`, driving force :math:`d_w`, and :math:`N=2,3`. Different initial
    conditions can be used, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{(x_i-0.5)^2 + (y_j-0.5)^2}}{\sqrt{2}\varepsilon}\right),

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is treated
    *semi-implicitly*, i.e., the linear part is solved with Fast-Fourier Transform (FFT) using ``mpi4py-fft`` [1]_ and the nonlinear part in the right-hand
    side will be treated explicitly.
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        f.impl[:] = self._eval_Laplacian(u, f.impl)

        if self.spectral:

            tmp = newDistArray(self.fft, False)
            tmp[:] = self.fft.backward(u, tmp)

            if self.eps > 0:
                tmpf = -2.0 / self.eps**2 * tmp * (1.0 - tmp) * (1.0 - 2.0 * tmp)
            else:
                tmpf = self.dtype_f(self.init, val=0.0)

            # build sum over RHS without driving force
            Rt_local = float(self.xp.sum(self.fft.backward(f.impl) + tmpf))
            if self.comm is not None:
                Rt_global = self.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
            else:
                Rt_global = Rt_local

            # build sum over driving force term
            Ht_local = float(self.xp.sum(6.0 * tmp * (1.0 - tmp)))
            if self.comm is not None:
                Ht_global = self.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
            else:
                Ht_global = Ht_local

            # add/substract time-dependent driving force
            if Ht_global != 0.0:
                dw = Rt_global / Ht_global
            else:
                dw = 0.0

            tmpf -= 6.0 * dw * tmp * (1.0 - tmp)
            f.expl[:] = self.fft.forward(tmpf)

        else:

            if self.eps > 0:
                f.expl = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)

            # build sum over RHS without driving force
            Rt_local = float(self.xp.sum(f.impl + f.expl))
            if self.comm is not None:
                Rt_global = self.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
            else:
                Rt_global = Rt_local

            # build sum over driving force term
            Ht_local = float(self.xp.sum(6.0 * u * (1.0 - u)))
            if self.comm is not None:
                Ht_global = self.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
            else:
                Ht_global = Ht_local

            # add/substract time-dependent driving force
            if Ht_global != 0.0:
                dw = Rt_global / Ht_global
            else:
                dw = 0.0

            f.expl -= 6.0 * dw * u * (1.0 - u)

        self.work_counters['rhs']()
        return f
