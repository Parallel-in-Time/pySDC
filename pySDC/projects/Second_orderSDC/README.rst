Spectral Deferred Correction Methods for Second-Order Problems
==============================================================

Python code for implementing the paper's plots on Second-order SDC methods.

You are welcome to use and adapt this code under the terms of the BSD license.
If you utilize it, either in whole or in part, for a publication, please provide proper citation:

.. code-block:: tex

    @misc{akramov2023spectral,
        title={Spectral deferred correction methods for second-order problems},
       author={Ikrom Akramov and Sebastian Götschel and Michael Minion and Daniel Ruprecht and Robert Speck},
       year={2023},
       eprint={2310.08352},
       archivePrefix={arXiv},
       primaryClass={math.NA}}


Reproducing Figures from the Publication
----------------------------------------

- **Fig. 1:** Execute `harmonic_oscillator_run_stability.py` while setting `kappa_max=18` and `mu_max=18`.
- **Fig. 2:** Run `harmonic_oscillator_run_stability.py` with the following configurations:
   - Set `kappa_max=30` and `mu_max=30`.
   - Adjust `maxiter` to 1, 2, 3, or 4 and execute each individually.
- **Table 1:** Execute `harmonic_socillator_run_stab_interval.py`:
   - To save the results set: `save_file=True`

- Use the script `harmonic_oscillator_run_points.py` to create a table based on given $(\kappa, \mu)$ points. This table assists in determining suitable values for `M`, `K`, and `quadrature nodes` to ensure stability in the SDC method.
   - To save the results set: `save_file=True`

- **Fig. 3:** Run `penningtrap_run_error.py` (Run local convergence: `conv.run_local_error()`) with `dt=0.015625/4` and `axes=(0,)`.
- **Fig. 4:** Run `penningtrap_run_error.py` (Run local convergence: `conv.run_local_error()`) using `dt=0.015625*4` and `axes=(2,)`.
- **Fig. 5:** Run `penningtrap_run_error.py` (Run global convergence: `conv.run_global_error()`) with `dt=0.015625*2`:
   - Note: Perform each run individually: first with `axes=(0,)`, then with `axes=(2,)`.
   - Manually set y-axis limits in `penningtrap_run_error.py`, specifically in lines 147-148.
- **Table 2:** Execute `penningtrap_run_error.py` (Run global convergence: `conv.run_global_error()`) with the following settings:
   - Expected order and approximate order are saved in the file: `data/global_order_vs_approx_order.csv`
   - Set: `K_iter=(2, 3, 4, 10)`
   - For `M=2`:
      - Set `dt=0.015625 / 2` and `num_nodes=2` (Both Horizontal and Vertical axes)
   - For `M=3`:
      - Set `dt=0.015625` and `num_nodes=3` (Both Horizontal and Vertical axes)
   - For `M=4`:
      - Set `dt=0.015625 * 4` and `num_nodes=4` (Both Horizontal and Vertical axes)
- **Fig. 6:** Run `penningtrap_run_Hamiltonian_error.py`:
   - Note: Execute each computation individually in case of crashes.
   - Data is saved in the data folder.
- **Fig. 7:** Execute `penningtrap_run_work_precision.py` with the following configurations:
   - Set `dt=0.015625*2` and `K_iter=(1, 2, 3)` for `axis=(2,)`.
   - Set `dt=0.015625*4`, and `K_iter=(2, 4, 6)`, and `dt_cont=2` for `axis=(0,)`.


Contact
-------
This code is written by `Ikrom Akramov <https://www.mat.tuhh.de/home/iakramov/?homepage_id=iakramov)>`_
