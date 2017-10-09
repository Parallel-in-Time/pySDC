from setuptools import setup, find_packages

with open("README.rst", 'r') as f:
    long_description = f.read()

setup(
    name='pySDC',
    version='2.0',
    description='A Python implementation of spectral deferred correction methods and the likes',
    license="BSD-2",
    long_description=long_description,
    author='Robert Speck',
    author_email='r.speck@fz-juelich.de',
    url="http://www.parallelintime.org/pySDC/",
    download_url = "https://github.com/Parallel-in-Time/pySDC/",

    packages=find_packages(exclude=['data', 'cover', 'docs', 'tests*']),

    package_data={
         '': ['*.txt', '*.rst'],
     },

    install_requires=[
        'nose>=1.3.7',
        'numpy>=1.9.3',
        'scipy>=0.17.1',
        'future>=0.15.2',
        'matplotlib>=1.5.3',
        'coloredlogs',
        'pep8',
        'sympy>=1.0'
    ],
)