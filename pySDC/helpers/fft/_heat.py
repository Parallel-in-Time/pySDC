"""
A distributed transform backed by Heat, as an alternative to owning one.

``pySDC.helpers.fft`` exists because released ``mpi4py-fft`` cannot drive a GPU, and carries some
seven hundred lines of distributed FFT that are not what pySDC is for. `Heat
<https://github.com/helmholtz-analytics/heat>`_ has that already: a distributed array over PyTorch
tensors, with FFTs that transpose across ranks when the transform axis is the split one.

Nothing here is a datatype. A ``DNDarray`` is built from the local block with ``is_split``, so no
data is gathered, and ``torch`` and ``cupy`` share device memory both ways without copying --
measured, both pointers equal. The arrays pySDC passes in are the arrays it gets back, and its
problem classes, sweepers and datatypes are untouched.

What this does not do, and what it would cost, is in ``PFFT_Heat.__init__``.
"""

import numpy as np


class PFFT_Heat:
    """The part of :class:`mpi4py_fft.mpifft.PFFT` that pySDC calls, on top of Heat.

    Heat distributes over a single axis, so this is a slab decomposition: one axis is split and the
    rest are whole. ``mpi4py-fft`` can split two, which matters once the rank count passes the
    length of one dimension.
    """

    on_GPU = True

    def __init__(
        self,
        comm,
        shape=None,
        axes=None,
        dtype=float,
        grid=None,
        padding=False,
        collapse=False,
        backend='heat',
        transforms=None,
        comm_backend=None,
        **kw,
    ):
        import heat as ht

        if transforms:
            # The spectral helper hands in a DCT for a Chebyshev or ultraspherical axis. Those axes
            # are never distributed -- `SpectralHelper1D.distributable` is False and `get_pfft`
            # orders them first -- so they could be applied locally here before handing the Fourier
            # axes to Heat. Doing so needs the transform order unpicked from `mpi4py-fft`'s
            # convention, which is the bulk of what is missing.
            raise NotImplementedError('per-axis transforms are not wired up yet; see the module docstring')
        if padding is not False:
            # `ht.fft.fftn` takes `s`, so this is a matter of passing the padded lengths through
            raise NotImplementedError('padding for dealiasing is not wired up yet')

        self.comm = comm
        self.ht = ht
        self._shape = list(shape)
        self._dtype = np.dtype(dtype)
        self._real = np.issubdtype(self._dtype, np.floating)

        ndim = len(self._shape)
        self._axes = tuple(range(ndim)) if axes is None else tuple(a % ndim for a in np.ravel(axes))

        # Heat splits one axis. Pick the one mpi4py-fft would leave whole last, so that the axis
        # transformed first is local, as its convention requires.
        self._split = 0 if self._axes[-1] != 0 else 1 % ndim

        # Heat's real transform halves the first of `axes`; NumPy and mpi4py-fft halve the last,
        # and the problem classes build their wavenumbers for that layout. The transforms are
        # separable so the order is free, and reversing puts the halved axis where pySDC expects.
        self._fft_axes = self._axes[::-1] if self._real else self._axes

        # Throwaway arrays are the honest way to learn both the decomposition and the output
        # shape: asking Heat beats reimplementing its chunking and its choice of which axis a real
        # transform halves, and hoping the two agree. `mpi4py-fft` plans with a throwaway transform
        # for the same reason.
        probe = ht.zeros(tuple(self._shape), split=self._split, device='gpu')
        self._local_shape = tuple(probe.lshape)
        self._offset = self._offset_of(probe)

        if self._real:
            probe = probe.astype(ht.float32 if self._dtype == np.dtype('float32') else ht.float64)
        else:
            probe = probe.astype(ht.complex64 if self._dtype == np.dtype('complex64') else ht.complex128)

        probed = (ht.fft.rfftn if self._real else ht.fft.fftn)(probe, axes=self._fft_axes)
        self._out_shape = list(probed.shape)
        self._out_local_shape = tuple(probed.lshape)
        self._out_split = probed.split
        self._out_offset = self._offset_of(probed)
        self._out_dtype = np.dtype(str(probed.larray.dtype).replace('torch.', ''))

    @staticmethod
    def _offset_of(array):
        """Where this rank's block starts along the split axis."""
        if array.split is None:
            return 0
        starts = np.cumsum([0] + [int(n) for n in array.lshape_map[:, array.split]])
        return int(starts[array.comm.rank])

    # -- the shape questions pySDC asks -------------------------------------------------------

    def global_shape(self, forward_output=False):
        return tuple(self._out_shape if forward_output else self._shape)

    def shape(self, forward_output=False):
        return self._out_local_shape if forward_output else self._local_shape

    def local_slice(self, forward_output=True):
        # `mpi4py-fft` defaults this to True while defaulting `global_shape` to False, and
        # `IMEX_Laplacian_MPIFFT.getLaplacian` relies on both -- it builds wavenumbers over the
        # real-space shape and slices them with the spectral slice. Matching the interface means
        # matching the inconsistency.
        here = [slice(0, s) for s in self.global_shape(forward_output)]
        split = self._out_split if forward_output else self._split
        offset = self._out_offset if forward_output else self._offset
        length = self.shape(forward_output)[split]
        here[split] = slice(offset, offset + length)
        return tuple(here)

    def dtype(self, forward_output=False):
        return self._out_dtype if forward_output else self._dtype

    def destroy(self):
        """Heat owns no communicators of its own here, so there is nothing to release."""

    # -- the transforms -----------------------------------------------------------------------

    def _wrap(self, u):
        """The local block as a distributed array, without moving it."""
        import torch

        return self.ht.array(torch.as_tensor(u), is_split=self._split)

    @staticmethod
    def _unwrap(a, out):
        """Hand the local block back as whatever came in, without moving it."""
        import cupy as cp

        result = cp.asarray(a.larray)
        if out is None:
            return result
        out[...] = result
        return out

    def forward(self, u, out=None, normalize=True):
        op = self.ht.fft.rfftn if self._real else self.ht.fft.fftn
        result = op(self._wrap(u), axes=self._fft_axes)
        if normalize:
            result = result / np.prod([self._shape[a] for a in self._axes])
        return self._unwrap(result, out)

    def backward(self, u, out=None, normalize=False):
        op = self.ht.fft.irfftn if self._real else self.ht.fft.ifftn
        kwargs = {'s': tuple(self._shape[a] for a in self._fft_axes)} if self._real else {}
        result = op(self._wrap(u), axes=self._fft_axes, **kwargs)
        if normalize:
            result = result * np.prod([self._shape[a] for a in self._axes])
        return self._unwrap(result, out)
