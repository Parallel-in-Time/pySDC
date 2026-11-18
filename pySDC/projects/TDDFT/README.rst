Real-time TDDFT with SDC parallel across the nodes
==================================================

This project compares SDC with the propagators of real-time time-dependent density functional theory (TDDFT) codes on
a model with the structure of a production pseudopotential code. The result: SDC whose node updates run in parallel
reaches a given accuracy of the dipole response with **3.5-5x less critical-path work** than the best production
propagator tested (commutator-free Magnus 4 with Lanczos exponentials), for relative dipole errors from 1e-3 to 1e-7.
Serial SDC does not beat it.

The model
---------

A one-dimensional molecule (two nuclei of charge 3, six electrons in three doubly occupied orbitals) in a 4-cycle,
800 nm laser pulse of 1e14 W/cm² (length gauge), on a pseudo-spectral grid (N = 1024, dx = 0.2) with a complex
absorbing potential:

.. math::
    i\partial_t\psi_j = -\tfrac12\partial_x^2\psi_j + \big(v_{loc} + v_H[\rho] + v_x[\rho] + xE(t) - i\,v_{cap}\big)
    \psi_j + \sum_a E_{kb}\,|\beta_a\rangle\langle\beta_a|\psi_j\rangle.

The ingredients that matter for the time integrator are those of production codes:

- several orbitals sharing one self-consistent density;
- a nonlocal Kleinman-Bylander projector on each atom;
- Hartree and exchange potentials of the density (here soft-Coulomb and the LDA exchange formula as a model
  nonlinearity);
- strong-field ionization: about 3% of the electrons leave and are absorbed.

The ground state comes from a self-consistent diagonalization. See ``problem.py``.

The methods
-----------

Cost is counted, not timed: the unit is one orbital-sized FFT round trip, i.e. one Hamiltonian application or one
kinetic solve per orbital. In 3D with many orbitals these dominate, and the density update (shared by all orbitals)
is counted separately.

- **SDC** (``sdc.py``, pySDC's ``imex_1st_order``). The kinetic part is implicit and solved exactly by FFT; its
  right-hand side is then known for free. The potential, projectors and laser are explicit. Gauss nodes, iterated
  until the collocation residual is below a tolerance.

  - The **serial** sweep (``QI='IE', QE='EE'``) costs ``1 + M k`` per step and orbital.
  - The **parallel** sweep (``QI='MIN-SR-S', QE='PIC'``) updates all M nodes independently. On M processes its
    critical path is ``1 + k``.

- **Commutator-free Magnus 4** (two exponentials per step, Lanczos, potentials extrapolated from the previous steps).
- **AETRS** with Taylor-4 exponentials (the default of Octopus).
- **Crank-Nicolson** (GMRES with the kinetic inverse as preconditioner).
- **Split-operator** Strang and triple jump, with exact exponentials of the projectors.

All of them are in ``propagators.py`` and use the same problem class and counters.

Results
-------

Full comparison (``python -m pySDC.projects.TDDFT.work_precision --full``): cost to reach a relative dipole error over
the pulse, against a tightly converged SDC reference.

=================================  =======  =======  =======  =======  =======
dipole error                       1e-3     1e-4     1e-5     1e-6     1e-7
=================================  =======  =======  =======  =======  =======
SDC, nodes in parallel (critical)  39 k     39 k     84 k     84 k     149 k
Magnus 4, Lanczos                  139 k    182 k    300 k    440 k    679 k
AETRS, Taylor 4                    269 k    538 k    --       --       --
SDC, serial                        153 k    222 k    365 k    --       --
split-operator 2                   27 k     --       --       --       --
=================================  =======  =======  =======  =======  =======

- **Parallel SDC.** 3.5x, 4.6x, 3.6x, 5.2x and 4.6x cheaper than Magnus 4 on the critical path. Over its 3-4
  processes the total work is about that of Magnus 4.
- **Split-operator** is the cheapest method at 1e-3, but stalls below it. In this strongly ionizing case the
  length-gauge laser makes the kinetic-potential commutator large for fast electrons, and the convergence drops to
  about first order.
- **Harmonic spectra** are converged to the plotted digit by every run with a dipole error below about 1e-3.

The figure is the small version the tests run (N = 256, 2 cycles, weaker field):

.. image:: ../../../data/tddft_work_precision.png
    :width: 100%

Iterating to a residual, not a fixed number of sweeps
------------------------------------------------------

The diagonal MIN-SR-S sweep is nilpotent in the stiff limit, but on the oscillatory (imaginary) spectrum of the
Schrödinger operator it first amplifies the error before it contracts. With a fixed number of sweeps, too few
diverges: in a single-orbital version, M = 3 needed at least 10 sweeps, and M = 4 diverged for every K up to 14.
Iterating until the residual is small passes through the growth. Then M = 4 is robust and the most efficient choice,
at 13-15 sweeps per step.

Carrying it over to 3D codes
----------------------------

- **Plane-wave codes** (diagonal kinetic operator): the same split applies directly. The kinetic solve is an exact
  FFT division, and nonlocal pseudopotentials and the density-dependent potentials are explicit. The parallel sweep
  adds a factor M of parallelism on top of the orbital and domain parallelism, which saturates in strong scaling.
- **Real-space finite-difference codes:** the kinetic solve becomes an iterative one, which would cost part of the
  gain. Not tested here.
- **Not covered:** the model exchange potential is not a 1D functional, there is no ionic motion, and only the
  strong-field (laser) regime was compared, not linear-response spectra over long times.

Running it
----------

``python -m pySDC.projects.TDDFT.work_precision`` runs the small comparison in a few minutes and writes
``data/tddft_work_precision.png``. ``--full --procs 3`` runs the full one (several hours); finished runs are kept in
``data/tddft_full/runs.json`` and skipped when restarted. The tests (``tests/test_tddft.py``) check the ground state,
the orders of the propagators, that both SDC sweeps converge to the same collocation solution, and the small
comparison.
