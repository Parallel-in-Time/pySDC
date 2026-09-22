# The CuPy stub lives with the main test suite, but this project's `cupy`-marked tests want it too;
# `pySDC/tests/conftest.py` only applies under `pySDC/tests`. See `pySDC/tests/fake_cupy.py`.
from pySDC.tests.conftest import *  # noqa: F401,F403
