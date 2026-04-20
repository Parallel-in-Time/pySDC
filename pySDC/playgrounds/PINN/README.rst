PINN playground with DeepXDE
============================

This playground starts a minimal setup for experiments that combine
`DeepXDE <https://github.com/lululxvi/deepxde>`_ with pySDC ideas.

Current toy example
-------------------

The script ``deepxde_toy_ode.py`` trains a PINN for

.. math::

    y'(t) - y(t) = 0, \quad t \in [0, 1], \quad y(0)=1,

whose exact solution is :math:`y(t)=\exp(t)`.

It is intentionally small and fast, so it can serve as a starting point
for later SDC-coupled prototypes.

Setup with micromamba
---------------------

From this directory:

.. code-block:: bash

    micromamba env create -f environment.yml
    micromamba activate pySDC_pinn

The environment installs:

- pySDC in editable mode from the repository root,
- DeepXDE with a PyTorch backend,
- basic numerical and plotting dependencies.

Run the toy problem
-------------------

.. code-block:: bash

    python deepxde_toy_ode.py

Expected output:

- training progress from DeepXDE,
- relative L2 error against the exact solution,
- a plot at ``data/deepxde_toy_ode_solution.png``.

Second case: ROBER without QSSA
-------------------------------

The script ``deepxde_rober_no_qssa.py`` solves the full stiff ROBER system
with three species and no quasi-steady-state reduction.

The collocation points are sampled logarithmically in time to capture both
fast initial transients and slower long-time behavior (default
``t in [1e-5, 1e5]``).

.. code-block:: bash

    python deepxde_rober_no_qssa.py

You can reduce runtime while testing by lowering the number of epochs, e.g.

.. code-block:: bash

    python deepxde_rober_no_qssa.py --epochs 500

For a very quick smoke test, reduce both epochs and time horizon:

.. code-block:: bash

    python deepxde_rober_no_qssa.py --epochs 50 --t-max 1e2 --num-collocation 256 --num-eval 200

Expected artifacts:

- ``data/deepxde_rober_no_qssa_solution.png``
- ``data/deepxde_rober_no_qssa_metrics.txt``

Third case: Figure-4-style regular PINN for ROBER
-------------------------------------------------

The script ``deepxde_rober_regular_fig4.py`` follows the regular PINN
configuration described around Figure 4 in the stiff-PINN paper:

- full ROBER system (no QSSA in the equations),
- hard-coded IC architecture ``y(t)=y0+(t/t_scale)SNN(log(t/t_scale))``,
- 2500 residual points in ``t in [1e-5, 1e5]`` sampled uniformly in log scale,
- 3 hidden layers with 128 neurons, GELU activation, Adam with ``lr=1e-3`` and batch size ``128``,
- defaults that are closer to the paper/authors' code: log-time NN input, hard ICs, minibatching, and normalized time scaling inside the ansatz.

Run a paper-style training:

.. code-block:: bash

    python deepxde_rober_regular_fig4.py

Run a quick smoke test:

.. code-block:: bash

    python deepxde_rober_regular_fig4.py --iterations 200 --num-points 512 --num-eval 300

Enable guarded/minibatch training (useful for sweeps/debugging, not paper baseline):

.. code-block:: bash

    python deepxde_rober_regular_fig4.py --batch-size 128 --max-loss-stop 1e4 --max-divergence-loss 1e12

Compare paper ingredients directly:

.. code-block:: bash

    python deepxde_rober_regular_fig4.py --run-tag paper_like
    python deepxde_rober_regular_fig4.py --no-hard-ic --run-tag no_hard_ic
    python deepxde_rober_regular_fig4.py --no-use-log-input --run-tag no_log_input

Run a broader parameter sweep with automatic ranking:

.. code-block:: bash

    python deepxde_rober_regular_fig4_sweep.py --iterations 4000 --max-loss-stop 1e4 --sweep-tag fig4_scan

Sweep outputs are written to ``data/`` as tagged logs, metrics, and summary files.

Expected artifacts:

- ``data/deepxde_rober_regular_fig4_solution.png``
- ``data/deepxde_rober_regular_fig4_loss.png``
- ``data/deepxde_rober_regular_fig4_metrics.txt``

Fourth case: upstream-style Stiff-PINN Robertson with QSSA
----------------------------------------------------------

The script ``stiff_pinn_robertson_qssa.py`` ports the Robertson QSSA example
from the upstream `DENG-MIT/Stiff-PINN <https://github.com/DENG-MIT/Stiff-PINN>`_
repository into a portable playground runner:

- keeps the PyTorch hard-IC ansatz used by the upstream QSSA model,
- reconstructs the eliminated intermediate species through the QSSA formula,
- uses SciPy BDF for the reference solution instead of the upstream ``assimulo`` dependency,
- writes plots, model weights, and metrics into ``data/``.

Run a quick Robertson QSSA training:

.. code-block:: bash

    python stiff_pinn_robertson_qssa.py --epochs 400 --run-tag smoke

Run a longer training closer to the upstream setup:

.. code-block:: bash

    python stiff_pinn_robertson_qssa.py --epochs 2000 --batch-size 512 --run-tag long

Expected artifacts:

- ``data/stiff_pinn_robertson_qssa_solution.png``
- ``data/stiff_pinn_robertson_qssa_model.pt``
- ``data/stiff_pinn_robertson_qssa_metrics.txt``

Fifth case: simple DeepXDE regular PINN matching paper setup
------------------------------------------------------------

The script ``deepxde_rober_paper_simple.py`` is a compact DeepXDE-only
reproduction of the regular (non-QSSA) ROBER setup described in the paper:

- full ROBER equations with ``k1=0.04, k2=3e7, k3=1e4``,
- logarithmic time domain ``t in [1e-5, 1e5]``,
- 2500 residual points sampled uniformly in logarithmic scale,
- hard-IC ansatz ``y=y0+(t/t_scale)SNN(log(t/t_scale))``,
- 3 hidden layers with 128 neurons (GELU),
- Adam with ``lr=1e-3`` and minibatch size ``128``.

It now supports two approaches:

- ``--approach global``: original single-network baseline over the full time window,
- ``--approach slab_irk``: sequential local slabs, each with a local PINN and implicit RK guide points.

For slab mode, implicit one-step guide methods are available via ``--irk-order``:

- ``--irk-order 2``: implicit midpoint (order 2),
- ``--irk-order 4``: 2-stage Gauss-Legendre IRK (order 4).

Quick smoke run:

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode smoke --approach global

Quick slab+IRK2 smoke run:

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode smoke --approach slab_irk --irk-order 2 --num-slabs 8

Quick slab+IRK4 smoke run:

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode smoke --approach slab_irk --irk-order 4 --num-slabs 8

Compare IRK orders directly (runs both and prints RMSE lines):

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode smoke --approach slab_irk --num-slabs 8 --compare-irk-orders

Run a small IRK sweep over order/slabs/steps and rank by mean RMSE:

.. code-block:: bash

    python run_deepxde_rober_paper_simple_irk_sweep.py --mode smoke --num-slabs-list 4,8 --steps-per-slab-list 20,40 --irk-weight-list 1.0 --seed-list 42

Paper-style run:

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode paper --approach global

Paper-style slab+IRK4 run:

.. code-block:: bash

    python run_deepxde_rober_paper_simple.py --mode paper --approach slab_irk --irk-order 4 --num-slabs 8

Expected artifacts:

- ``data/deepxde_rober_paper_simple_solution.png``
- ``data/deepxde_rober_paper_simple_loss.png``
- ``data/deepxde_rober_paper_simple_metrics.txt``

Per run (for tagged runs), additional variable-wise plots are written:

- ``data/deepxde_rober_paper_simple_<tag>_y1.png``
- ``data/deepxde_rober_paper_simple_<tag>_y2.png``
- ``data/deepxde_rober_paper_simple_<tag>_y3.png``

