import os

# Run the `cupy`-marked tests against a NumPy-backed stand-in for CuPy, for machines with no
# GPU. See `pySDC/tests/fake_cupy.py` for what this does and does not actually test.
if os.environ.get('PYSDC_FAKE_GPU', '') not in ('', '0'):
    from pySDC.tests import fake_cupy

    fake_cupy.install()

    def pytest_report_header(config):
        return 'PYSDC_FAKE_GPU is set: cupy is stubbed out with numpy, NO GPU code is being run'
