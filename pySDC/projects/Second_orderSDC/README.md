# Spectral Deferred Correction methods for Second-order problems

Python code to implement Second-order SDC paper plots.

## Attribution
You can freely use and reuse this code in line with the BSD license. 
If you use it (or parts of it) for a publication, please cite:

- [Publication citation information will be added soon]

[![DOI](http://example.com)](http://example.com)

## How can I reproduce Figures from the publication?

- Fig. 1: Run `dampedharmonic_oscillator_run_stability.py`.
- Fig. 2: Run `dampedharmonic_oscillator_run_stability.py` with the following settings:
   - Set `maxiter` to 1, 2, 3, or 4 and run individually.
- Fig. 3: Run `penningtrap_run_error.py` (Run local convergence) with `dt=0.015625/4` and `axes=[0]`.
- Fig. 4: Run `penningtrap_run_error.py` (Run local convergence) with `dt=0.015625*4` and `axes=[2]`.
- Fig. 5: Run `penningtrap_run_error.py` (Run global convergence) with `dt=0.015625*2` and `axes=[0]` and `axes=[2]`.
   - Note: Each run needs to be performed individually.
   - The y-axis limits should be manually set in `penningtrap_run_error.py`, specifically in lines 103-104.
- Fig. 6: Run `penningtrap_run_work_precision.py` with the following settings:
   - Set `dt=0.015625*2` and `K_iter=[1, 2, 3]` in `axis=[2]`.
   - Set `K_iter=[2, 4, 6]` in `axis=[0]`.

## Who do I talk to?

This code is written by [Ikrom Akramov](https://www.mat.tuhh.de/home/iakramov/?homepage_id=iakramov).
