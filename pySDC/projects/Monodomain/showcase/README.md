# Can SDC beat the state of the art for the cardiac monodomain equation? (closed)

**Final verdict, November 2026: no.** Exponential SDC (ESDC) beats the Rush-Larsen schemes that cardiac codes use on
counted work. On real hardware it does not.

## The question

The monodomain equation couples diffusion of the membrane potential to a stiff ionic cell model, here ten
Tusscher-Panfilov with 19 variables per cell. Cardiac codes integrate it with Rush-Larsen schemes: the gating
variables are updated exactly for rates frozen over the step, the rest explicitly, and diffusion implicitly. RL1 is
the first-order default; RL2 is the second-order Strang version.

We compared ESDC against them. Accuracy was measured as the error in activation times (when the potential first
crosses -20 mV), cost as counted work.

## What happened

| stage | report | result |
|---|---|---|
| 1. serial ESDC | [stage1_gate.md](stage1_gate.md) | Wins only below about 0.2 ms activation error, far below the 2-7 ms spatial error of realistic meshes. |
| follow-up | [stability_and_local_adaptivity.md](stability_and_local_adaptivity.md) | Stability is not the limit, resolving the upstroke is. Local adaptivity gives 10-140x, but helps the low-order schemes more. |
| 2. nodes in parallel | [stage2_parallel_nodes.md](stage2_parallel_nodes.md) | Counted on the critical path, it wins at every accuracy, 1.7-3.5x at 0.1-1 ms. |
| real hardware | [stage2_parallel_nodes.md](stage2_parallel_nodes.md) | One T4 batches 4 nodes at half price only up to 128² cells. On 4 JUSUF cores it is only as fast as RL2 on 1 core (1.06-1.13x). |

## Why it fails

- **The field needs only low accuracy.** Spatial error dominates, so time steps sit where low-order schemes are
  cheapest.
- **ESDC pays for its exponential quadrature.** Every cell has its own rates, so every cell carries its own
  quadrature weights, which have to be streamed through memory every sweep. In the profile, that cost more than the
  ionic model (54% of the time against 31%), which is a cost Rush-Larsen does not have. The counting model of stages
  1-2 ignored it.
- **Parallelism across nodes needs spare devices.** Per GPU the ionic kernel fills the device at a few thousand
  cells. On CPUs the per-step setup is repeated on every rank, so parallel efficiency is 49-56%.

## Kept as useful regardless

- The RL1/RL2 sweepers (`sweeper_classes/runge_kutta/rush_larsen.py`) and the per-kernel work counters.
- The node-parallel ESDC sweepers (`imexexp_parallel.py`, and `imexexp_parallel_MPI.py` with one node per MPI
  rank). They converge as fast as the serial sweep when stopped on the residual.
- Two fixes in the ESDC sweeper: its phi cache now also depends on dt (restarts reused stale phi), and its `allclose`
  on meshes no longer crashes on NaN.
