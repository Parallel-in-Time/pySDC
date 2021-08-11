from setuptools import setup, find_packages

with open("README.rst", 'r') as f:
    long_description = f.read()

setup(
    name='pySDC',
    version='4.1',
    description='A Python implementation of spectral deferred correction methods and the likes',
    license="BSD-2",
    long_description=long_description,
    author='Robert Speck',
    author_email='r.speck@fz-juelich.de',
    url="http://www.parallel-in-time.org/pySDC/",
    download_url="https://github.com/Parallel-in-Time/pySDC/",

    packages=find_packages(),

    include_package_data=True,

    install_requires=[
        'numpy>=1.15.4',
        'scipy>=0.17.1',
        'matplotlib>=3.0',
        'sympy>=1.0',
        'numba>=0.35',
        'dill>=0.2.6',
    ],
)
