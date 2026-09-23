"""
A NumPy/SciPy-backed stand-in for CuPy, so the ``useGPU=True`` code paths can be exercised
on a machine without a GPU.

This is a *smoke test* facility, not a substitute for running on real hardware. It installs
fake ``cupy``, ``cupyx`` and ``cupy_backends`` modules into ``sys.modules`` that forward to
NumPy and SciPy. What that does and does not buy you:

Covered
    Everything that is about plumbing rather than about CUDA: that ``setup_GPU`` populates
    all the class attributes its CPU twin does, that the GPU problem/transfer/sweeper classes
    still accept the arguments the tests pass them, that ``xp`` and ``sparse_lib`` are not
    mixed up, and that the GPU modules import at all. Since nearly every ``*_GPU`` test is
    the corresponding CPU test called with ``useGPU=True``, that is most of them.

Not covered
    Anything that actually needs a device: CUDA kernels, ``DistArrayCuPy``, the
    ``CuSparseError`` out-of-memory fallbacks, GPU timings, and any place where CuPy's
    behaviour genuinely differs from NumPy's. NCCL cannot be faked at all, because its calls
    take raw device pointers -- tests that build an ``NCCLComm`` skip on ``ACTIVE`` instead.
    A green run here does **not** mean the GPU code works.

Activated by setting ``PYSDC_FAKE_GPU=1``; see ``pySDC/tests/conftest.py``.
"""

import os
import sys
import types

import numpy
import scipy.fft
import scipy.sparse
import scipy.sparse.linalg

#: Whether the stub is switched on for this run.
ACTIVE = os.environ.get('PYSDC_FAKE_GPU', '') not in ('', '0')


def _no_nccl(*args, **kwargs):
    raise NotImplementedError(
        'NCCL cannot be stubbed out on the CPU: its calls take raw device pointers. Tests that '
        'need an NCCLComm have to skip when `pySDC.tests.fake_cupy.ACTIVE`.'
    )


def _module(name, **attrs):
    """Register an empty stand-in module under `name`."""
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__name__ = name  # `attrs` may have carried another module's name in
    sys.modules[name] = module
    return module


class _DeviceArray(numpy.ndarray):
    """A NumPy array that answers `.get()`, the way a CuPy array on a device does."""

    def get(self):
        return numpy.asarray(self).view(numpy.ndarray)

    def copy(self, *args, **kwargs):
        """Degrade to the base array, the way CuPy does.

        CuPy's `ndarray` is a Cython extension type whose `copy` returns the base class rather
        than `type(self)`, so a copy of a subclass silently stops being one. NumPy preserves the
        subclass, so this has to be imitated deliberately or the stub would be more forgiving
        than the hardware.
        """
        return numpy.ndarray.copy(self, *args, **kwargs).view(_DeviceArray)

    def __deepcopy__(self, memo=None):
        """CuPy routes `copy.deepcopy` through `copy`, and so loses the subclass there too."""
        return self.copy()


def _as_device(obj):
    return obj.view(_DeviceArray) if type(obj) is numpy.ndarray else obj


def _wrap(func):
    """Make `func` hand back `_DeviceArray`s, so results keep answering `.get()`."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return tuple(map(_as_device, result)) if type(result) is tuple else _as_device(result)

    wrapper.__name__ = getattr(func, '__name__', 'wrapped')
    wrapper.__doc__ = getattr(func, '__doc__', None)
    return wrapper


def _mirror(name, source):
    """Register a module under `name` exposing everything in `source`, functions wrapped.

    Classes are passed through unwrapped so that `isinstance` checks against them still work.
    """
    module = types.ModuleType(name)
    for key, value in source.__dict__.items():
        module.__dict__[key] = _wrap(value) if callable(value) and not isinstance(value, type) else value
    module.__name__ = name  # otherwise tracebacks claim this is numpy/scipy, which is confusing
    sys.modules[name] = module
    return module


class _Stream:
    def record(self, *args, **kwargs):
        pass

    def synchronize(self, *args, **kwargs):
        pass


class _Device:
    def synchronize(self):
        pass

    def use(self):
        pass


def install():
    """Put the fake CuPy modules into `sys.modules`. Safe to call more than once."""
    if isinstance(sys.modules.get('cupy'), types.ModuleType) and getattr(sys.modules['cupy'], '_pySDC_fake', False):
        return

    cuda = _module(
        'cupy.cuda',
        nccl=_module('cupy.cuda.nccl', get_unique_id=_no_nccl, NcclCommunicator=_no_nccl),
        Event=_Stream,
        Device=_Device,
        get_current_stream=lambda: _Stream(),
        get_elapsed_time=lambda before, after: 0.0,
        runtime=_module('cupy.cuda.runtime', getDeviceCount=lambda: 1),
    )

    cupy = _mirror('cupy', numpy)
    cupy.ndarray = _DeviceArray
    cupy.cuda = cuda
    cupy.asnumpy = lambda obj: numpy.asarray(obj).view(numpy.ndarray)
    cupy._pySDC_fake = True
    for submodule in ('fft', 'linalg', 'random'):
        setattr(cupy, submodule, _mirror(f'cupy.{submodule}', getattr(numpy, submodule)))

    cupyx = _module('cupyx')
    cupyx.scipy = _module('cupyx.scipy')
    cupyx.scipy.fft = _mirror('cupyx.scipy.fft', scipy.fft)
    # `import a.b.c` goes through the parent module's attribute rather than `sys.modules`, so
    # every level has to be hung off its parent or the real SciPy module wins.
    # Copied across unwrapped: pySDC builds sparse matrices with these and compares them
    # against ones SciPy made, so the types have to stay identical.
    sparse = _module('cupyx.scipy.sparse', **scipy.sparse.__dict__)
    sparse.linalg = _mirror('cupyx.scipy.sparse.linalg', scipy.sparse.linalg)
    cupyx.scipy.sparse = sparse

    backends = _module('cupy_backends')
    backends.cuda = _module('cupy_backends.cuda')
    backends.cuda.libs = _module('cupy_backends.cuda.libs')
    backends.cuda.libs.cusparse = _module(
        'cupy_backends.cuda.libs.cusparse', CuSparseError=type('CuSparseError', (Exception,), {})
    )

    # `.get()` on a cupyx sparse matrix hands back the SciPy one; here it already is one.
    # This patches SciPy itself, which is why the stub is opt-in rather than always on.
    for cls in (scipy.sparse.spmatrix, scipy.sparse.sparray):
        cls.get = lambda self: self

    _redirect_mpi4py_fft_backends()


def _redirect_mpi4py_fft_backends():
    """Point mpi4py-fft's CuPy backends at its NumPy/SciPy ones.

    Since `cupy` is NumPy here, that is the faithful translation, and it means the tests do not
    need an mpi4py-fft built with GPU support. The cost is that `DistArrayCuPy` is never taken:
    those distributed GPU arrays are one more thing only real hardware exercises.
    """
    try:
        from mpi4py_fft import libfft
    except ImportError:
        return

    fallbacks = {'cupy': 'numpy', 'cupyx-scipy': 'scipy'}
    original = libfft.FFT.__init__

    def __init__(self, *args, backend='fftw', **kwargs):
        original(self, *args, backend=fallbacks.get(backend, backend), **kwargs)

    libfft.FFT.__init__ = __init__
