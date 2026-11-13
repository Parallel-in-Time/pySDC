SDC for stochastic differential equations
=========================================

This project implements spectral deferred corrections (SDC) for stochastic differential equations (SDEs), following
Binkowski, Christian, Götschel and Minion: *Toward Spectral Deferred Correction Methods for Stochastic Differential
Equations* (`arXiv:2610.23693 <https://arxiv.org/abs/2610.23693>`_), and repeats the numerical experiments of that
paper with pySDC. The authors' MATLAB code is at `github.com/sgoetschel/SDC4SDEs
<https://github.com/sgoetschel/SDC4SDEs>`_.

The method
----------

An SDE :math:`dX_t = a(t, X_t)\, dt + \sum_r b_r(t, X_t)\, dW^r_t` is driven by Wiener processes (Brownian motions)
:math:`W^r`, whose paths are continuous but nowhere differentiable. The standard schemes, Euler-Maruyama and Milstein,
use only the increments :math:`\Delta W` of the paths over each time step, and converge with low order: the mean
error of single paths (the *strong* error) decays like :math:`\Delta T^{1/2}` and :math:`\Delta T` with the step
size :math:`\Delta T`, and the error in the expectation (the *weak* error) like :math:`\Delta T`.

SDC-SBB (SDC with smooth Brownian bridges) instead replaces the path on each time step by a smooth function
:math:`B^{(J)}` through its values at both ends of the step: the straight line between them plus :math:`J` terms of
the sine series of the remainder (the Karhunen-Loève expansion of a Brownian bridge). With :math:`dB^{(J)}/dt` in
place of the white noise :math:`dW/dt`, the SDE becomes an ordinary differential equation with random coefficients
on each step, the *random ODE*. Its solution converges to that of the SDE in Stratonovich form (Wong-Zakai theorem),
so the SDEs are written in Stratonovich form here, :math:`dX_t = \underline{a}(t, X_t)\, dt + \sum_r b_r(t, X_t)
\circ dW^r_t`, whose drift :math:`\underline{a} = a - \frac{1}{2} \sum_r (b_r \cdot \nabla) b_r` differs from the
Itô form above.

The random ODE is then solved with explicit SDC on :math:`M` Lobatto nodes per step. Because
:math:`dB^{(J)}/dt` grows like :math:`\Delta T^{-1/2}`, each sweep gains only half an order, and the order of the
collocation method, :math:`2M - 2` for ODEs, is halved too. "SDC-M" below is SDC-SBB on :math:`M` nodes with
:math:`2(M - 1)` sweeps, which converges with strong and weak order :math:`M - 1`.

With the straight line alone (:math:`J = 0`), the random ODE has the exact solution of the SDE at the ends of the
steps for SDEs of the form :math:`dX = \alpha b(X)\, dt + b(X) \circ dW`, which includes the first three examples
below. Other SDEs need the sine terms, whose fast oscillations the sweeper integrates exactly against the polynomial
through the diffusion :math:`b_r` at the nodes.

Usage
-----

The problem classes hold a whole ensemble of Brownian paths and solve for all of them at once: the solution has one
row per path. The sweeper computes the bridges of each step from the paths.

.. code-block:: python

    from pySDC.projects.SDC4SDEs.brownian import brownian_paths
    from pySDC.projects.SDC4SDEs.problem_classes import NonlinearScalarSDE
    from pySDC.projects.SDC4SDEs.run import errors, solve_sdc

    paths = brownian_paths(n_samples=1000, n_fine=1024, seed=0)  # Brownian paths on 1024 steps of [0, 1]
    u, P = solve_sdc(NonlinearScalarSDE, {'brownian_path': paths}, dt=1 / 32, num_nodes=4, num_sweeps=6)
    strong, weak = errors(u, P)  # against the exact solution for the same paths

``solve_sdc`` sets up an ordinary pySDC run with ``SBBSweeper``, Lobatto nodes, ``maxiter`` sweeps and no residual
tolerance.

Organisation of the project
---------------------------

- ``brownian.py``: Brownian paths, the coefficients of the smooth bridge on a step, and smoothed paths
- ``problem_classes.py``: the base class ``SDEProblem`` and the four examples of the paper, with their exact solutions
- ``sweeper.py``: ``SBBSweeper``, explicit SDC for the random ODE, with three initial guesses: ``'spread'``
  (the initial value at all nodes), ``'euler'`` (explicit Euler for the random ODE) and ``'euler_maruyama'`` (one
  Euler-Maruyama step for the Itô form, interpolated linearly to the nodes)
- ``baselines.py``: Euler-Maruyama, and Milstein for commutative noise
- ``run.py``: solving with any of these methods, and the strong and weak errors
- ``studies.py``: the experiments of the paper, with plots

Results
-------

Run all experiments with the sample sizes of the paper (10\ :sup:`4` paths; about 7 minutes on a laptop) by

.. code-block:: bash

    python pySDC/projects/SDC4SDEs/studies.py

which writes the plots into ``data/``. All methods see the same paths, and the errors are measured at :math:`T = 1`
against the exact solution for these paths: the strong error is the mean absolute error over the paths, and the weak
error the absolute difference of the means, as in the paper. Work is the number of evaluations of the drift, the
diffusion and, for Milstein, its derivative, per path.

- **Order and sweeps** (paper Figure 1, geometric Brownian motion, :math:`M = 5`): the initial guess has strong order
  0 (Euler) or 1/2 (Euler-Maruyama), and every second sweep adds one order, up to 4 after 8 sweeps.
- **Convergence and work** (paper Figures 2 to 6): SDC-3, SDC-4 and SDC-5 converge with order 2, 3 and 4, strongly
  and weakly, for geometric Brownian motion, the nonlinear SDE of Tian and Burrage and the linear system with two
  noises; except for large errors, they need fewer evaluations than Euler-Maruyama and Milstein. On the
  nonlinear SDE, the explicit Euler-Maruyama and Milstein schemes blow up on some paths for large steps, where the
  paper uses implicit versions of them; its steps therefore start at :math:`2^{-2}`.
- **Ornstein-Uhlenbeck process** (paper Figure 7, :math:`M = 4`, :math:`J = 128`): its noise is additive, so the
  straight line alone gives only order 1, and the sine terms are needed. As in the paper, the paths are first
  smoothed (64 global sine terms). SDC-4 then converges to the solution for the smooth paths, with errors that agree
  with those of the paper down to :math:`5 \cdot 10^{-5}` at :math:`\Delta T = 2^{-5}`, and against the rough paths
  it levels off at the distance of :math:`1.3 \cdot 10^{-4}` between the two solutions (paper:
  :math:`1.4 \cdot 10^{-4}`).

  Two differences to the paper. First, the paper's convergence to the smooth reference levels off at
  :math:`1.5 \cdot 10^{-6}`, which it attributes to the bridge. Here it does not: SDC-4 reaches
  :math:`7 \cdot 10^{-10}`, and the exact solution of the random ODE converges to the smooth reference down to
  :math:`10^{-11}`. The floor comes from the MATLAB code computing the sine coefficients
  :math:`\int \cos(j \pi s / \Delta T)\, dW` with a left-point rule on the fine grid of the path; with that rule the
  same floor appears here, independent of :math:`\Delta T` and shrinking with the fine grid (:math:`1.2 \cdot
  10^{-5}` for :math:`2^{14}` and :math:`2.6 \cdot 10^{-6}` for :math:`2^{16}` fine steps, the paper has
  :math:`2^{17}`). ``brownian.py`` integrates them exactly for the piecewise linear path. Second, the paper reports
  a fitted order of 2.9 against the smooth reference; the errors do not follow a power law, though, but drop sharply
  once the steps resolve the smooth path (from :math:`10^{-3}` at :math:`2^{-4}` to :math:`7 \cdot 10^{-8}` at
  :math:`2^{-6}`), so a fitted order depends on the range of steps. In addition to the paper, SDC-4 is also run on the
  rough paths themselves: there it converges with order 1 only, with or without the sine terms, because the
  oscillations of the bridge on each step are not resolved by the nodes.

Not implemented: implicit or IMEX SDC for stiff SDEs, implicit Euler-Maruyama and Milstein, Gauss nodes (the sweeper
needs a node at the right end of the step), SDEs with non-commutative noise, and the paper's convergence analysis.

Tests
-----

All tests are in ``tests`` and run in about 15 seconds with

.. code-block:: bash

    pytest pySDC/projects/SDC4SDEs/tests

They check the bridge and its quadrature, that SDC-SBB converges to the exact solution of the random ODE, the orders
of all methods on all examples, the order gained per sweep, the work counts, and the Ornstein-Uhlenbeck findings, and
make the plots of the website at reduced sample sizes.
