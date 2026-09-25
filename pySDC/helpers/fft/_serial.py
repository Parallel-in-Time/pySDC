"""The serial transforms, done by CuPy, wrapped the way ``mpi4py-fft`` expects them."""

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
from mpi4py_fft.libfft import FFTBase

#: Transforms CuPy has no kernel for. Half precision is the one that matters: the
#: finite-difference problems already take a ``dtype``, so a ``float16`` field is reachable, and it
#: survives a transpose (NCCL carries the bytes) but cannot be transformed.
_UNTRANSFORMABLE = (np.dtype('float16'),)


class _Wrap:
    """Give a CuPy transform the input/output array interface ``mpi4py-fft`` calls through.

    The NumPy version of this assigns with ``array[...] = other``, which CuPy refuses across
    dtypes; ``cupy.copyto`` with unsafe casting is the equivalent, and is the only reason this
    cannot simply be imported from ``mpi4py_fft.libfft``.
    """

    def __init__(self, transform, input_array, output_array, normalisation=1, options=None):
        self._transform = transform
        self._input_array = input_array
        self._output_array = output_array
        self._M = normalisation
        self._opt = {} if options is None else options
        self.__doc__ = getattr(transform, '__doc__', None)

    input_array = property(lambda self: self._input_array)
    output_array = property(lambda self: self._output_array)
    xfftn = property(lambda self: self._transform)
    opt = property(lambda self: self._opt)
    M = property(lambda self: self._M)

    def __call__(self, *args, **kwargs):
        self._opt.update(kwargs)
        cp.copyto(self._output_array, self._transform(self.input_array, **self._opt), casting='unsafe')
        if abs(self._M - 1) > 1e-8:
            self._output_array *= self._M
        return self.output_array


class _OuterWrap:
    """The outward-facing transform, which takes and returns arrays rather than owning them."""

    def __init__(self, transform, input_array, output_array):
        self._transform = transform
        self._input_array = input_array
        self._output_array = output_array
        self.__doc__ = getattr(transform, '__doc__', None)

    input_array = property(lambda self: self._input_array)
    output_array = property(lambda self: self._output_array)
    xfftn = property(lambda self: self._transform)

    def __call__(self, input_array=None, output_array=None, **options):
        if input_array is not None:
            cp.copyto(self._input_array, input_array, casting='unsafe')
        self._transform(**options)
        if output_array is not None:
            cp.copyto(output_array, self._output_array, casting='unsafe')
            return output_array
        return self.output_array


def _plan(shape, axes, dtype, transforms):
    """Pick the forward and backward transform for these axes and wrap them."""
    if np.dtype(dtype) in _UNTRANSFORMABLE:
        raise NotImplementedError(
            f'CuPy has no FFT for {np.dtype(dtype)}. A field of that type can be distributed and '
            'redistributed, but has to be cast up before it is transformed.'
        )

    transforms = {} if transforms is None else transforms
    real_transform = False
    if tuple(axes) in transforms:
        # a basis that is not Fourier -- Chebyshev and ultraspherical hand in their own DCT here,
        # which is why no FFT-only library can stand in for this
        forward, backward = transforms[tuple(axes)]
    elif np.issubdtype(np.dtype(dtype), np.floating):
        # A real field has a conjugate-symmetric spectrum, so half of it is redundant and the
        # transform returns an array shorter by half along the last axis. The problem classes rely
        # on that -- they build their wavenumbers to match -- so it is not an optimisation to skip.
        forward, backward = cufft.rfftn, cufft.irfftn
        real_transform = True
    else:
        forward, backward = cufft.fftn, cufft.ifftn

    s = tuple(np.take(shape, axes))
    u = cp.empty(shape=shape, dtype=dtype)
    v = cp.array(forward(u, s=s, axes=axes))
    normalisation = np.prod(s)

    options = {'s': s, 'axes': axes, 'overwrite_x': True}
    return (
        _Wrap(forward, u, v, 1, dict(options)),
        _Wrap(backward, v, u, normalisation, dict(options)),
        real_transform,
    )


class CuPyFFT(FFTBase):
    """A serial transform over some axes of a CuPy array.

    :class:`mpi4py_fft.libfft.FFT` chooses its transform from a dict literal in its own body, so a
    new backend cannot be added from outside; everything else it does is in ``FFTBase`` and is
    inherited, including the padding and truncation that dealiasing relies on.
    """

    backend = 'cupyx-scipy'

    def __init__(self, shape, axes=None, dtype=float, padding=False, transforms=None, **kw):
        FFTBase.__init__(self, shape, axes, dtype, padding)

        self.fwd, self.bck, real_transform = _plan(self.shape, self.axes, self.dtype, transforms)
        u, v = self.fwd.input_array, self.fwd.output_array

        self.M = 1.0 / np.prod(np.take(self.shape, self.axes))
        # `FFTBase` guessed this from the dtype; `_plan` knows whether a real transform was
        # actually chosen, which it is not for a basis that brought its own
        self.real_transform = real_transform

        self.padding_factor = 1.0
        if padding is not False:
            self.padding_factor = padding[self.axes[-1]] if np.ndim(padding) else padding

        if abs(self.padding_factor - 1.0) > 1e-8:
            assert len(self.axes) == 1, 'padding is only defined for a single axis at a time'
            truncated = self._get_truncarray(shape, v.dtype)
            self.forward = _OuterWrap(self._forward, u, truncated)
            self.backward = _OuterWrap(self._backward, truncated, u)
        else:
            self.forward = _OuterWrap(self._forward, u, v)
            self.backward = _OuterWrap(self._backward, v, u)

    def _forward(self, **kw):
        normalize = kw.pop('normalize', True)
        self.fwd(None, None, **kw)
        self._truncation_forward(self.fwd.output_array, self.forward.output_array)
        if normalize:
            self.forward._output_array *= self.M
        return self.forward.output_array

    def _backward(self, **kw):
        normalize = kw.pop('normalize', False)
        self._padding_backward(self.backward.input_array, self.bck.input_array)
        self.bck(None, None, **kw)
        if normalize:
            self.backward._output_array *= self.M
        return self.backward.output_array

    def _get_truncarray(self, shape, dtype):
        """The un-padded array a forward transform truncates into."""
        axis = self.axes[-1]
        shape = list(shape)
        shape[axis] = int(np.round(shape[axis] / self.padding_factor))
        if self.real_transform:
            # only half the spectrum is stored, as in `_plan`
            shape[axis] = shape[axis] // 2 + 1
        return cp.zeros(shape, dtype=dtype)
