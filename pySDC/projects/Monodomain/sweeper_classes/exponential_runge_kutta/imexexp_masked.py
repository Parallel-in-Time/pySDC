import numpy as np
from scipy.ndimage import binary_dilation

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_parallel import imexexp_parallel


class _Tracked:
    """
    Per-cell array (K, nstates, ncells) with its squared column sums and per-cell scaled maximum kept up to date
    incrementally, so that the solver's relative norm and the active set cost work only where values changed.
    """

    def __init__(self, K, nstates, n, scale):
        self.A = np.zeros((K, nstates, n))
        self.sumsq = np.zeros((K, nstates))
        self.cellmax = np.zeros(n)
        self.scale = scale

    def set(self, cells, values):
        if cells is None:
            self.A[:] = values
            self.sumsq = np.sum(values**2, axis=2)
            self.cellmax = (np.abs(values) / self.scale[None, :, None]).max(axis=(0, 1))
            return
        self.sumsq -= np.sum(self.A[:, :, cells] ** 2, axis=2)
        self.A[:, :, cells] = values
        self.sumsq += np.sum(values**2, axis=2)
        self.cellmax[cells] = (np.abs(values) / self.scale[None, :, None]).max(axis=(0, 1))

    def rel_norm(self, u0norm, exact=False):
        sumsq = np.sum(self.A**2, axis=2) if exact else np.maximum(self.sumsq, 0.0)
        return np.max(np.mean(np.sqrt(sumsq) / u0norm[None, :], axis=1))


class imexexp_masked(imexexp_parallel):
    r"""
    Node-parallel exponential SDC that stops updating cells once they have converged.

    The sweep is the one of ``imexexp_parallel`` written in delta form: for every node m,

    .. math::
        u_m \leftarrow u_m + \delta_m, \qquad \delta_{m,V} = (I - \Delta t\, q_m A)^{-1} r_{m,V}, \qquad
        \delta_{m,i} = r_{m,i} \;(i \neq V),

    with :math:`r_m = u_0 + I_m(u) - u_m` the collocation residual. With ``mask='none'`` this is exactly
    ``imexexp_parallel``. Otherwise :math:`\delta` is set to zero on frozen cells, and the ionic model, the residual
    and the exponential quadrature (phi functions, quadrature weights) are evaluated only where something can have
    changed. Because the residual is the true collocation residual, freezing changes how fast the sweep converges,
    not what it converges to.

    Every state i of a cell is scaled by its RMS over the tissue at the step start, like the solver's relative norm.

    - **Step start.** The initial guess copies u0 to all nodes, so the residual is exactly
      :math:`\Delta t\, c_m f(u_0)` for the non-gating states and bounded by it for the gating ones
      (:math:`|\varphi_1| \le 1` for decaying gates). Cells where that bound is below ``theta * restol`` start frozen,
      carry the bound as their residual, and need no phi functions unless they are woken up later.
    - ``mask='res'``: a cell is active while its residual exceeds ``theta * restol``. A frozen cell's residual can only
      change if a neighbour's potential changed, so it is recomputed on the active set dilated by the Laplacian
      stencil; elsewhere the cached value stays exact. Stops on the solver's relative residual.
    - ``mask='inc'``: after the first sweep, a cell is active while its last increment (an estimate of its iteration
      error) exceeds ``theta * restol``, plus the neighbours of such cells, whose diffusion input changed. Frozen cells
      are not looked at again, so there is no residual halo. Stops when the relative norm of the increments is below
      ``restol``.
    - ``mask='err'``: like ``'res'``, but each cell gets its own tolerance: it is frozen once its residual is below
      ``max(theta * restol, kappa * e)``, with e an estimate of the cell's own time-discretization error in this step,
      and the step ends when no cell is active. There is no point in converging a cell far below its discretization
      error, which is largest at the upstroke, where most sweeps are spent. e is the defect of the degree M-1
      polynomial through u0, u_1, ..., u_{M-1} extrapolated to the last node, |u_M - q(1)|, an O(dt^M) quantity, so it
      overestimates the collocation error; ``kappa`` absorbs that. e only loosens the tolerance from sweep ``kmin`` on,
      so that a crude early iterate cannot freeze itself.

    Reactivation is exact for ``'res'`` and ``'err'``: a frozen cell's residual depends on its own state (unchanged),
    the stimulus (fixed within the step) and its neighbours' potential, so recomputing the residual on every
    neighbour of an updated cell after each sweep catches every cell that something can have woken up, for example
    resting tissue that the arriving front starts to depolarise during the iteration. ``'inc'`` only adds the
    neighbours of cells whose increment is still large.

    The diffusion solve is a global DCT and is done on the whole tissue; only its result is masked. f at the step
    start is taken from the previous step's last node instead of being re-evaluated.
    """

    def __init__(self, params, level):
        for k, v in dict(mask='res', theta=0.1, kappa=0.1, kmin=2).items():
            params.setdefault(k, v)
        super().__init__(params, level)
        assert self.params.mask in ['none', 'res', 'inc', 'err']
        M, c = self.coll.num_nodes, self.coll.nodes
        x = np.concatenate([[0.0], c[:-1]])  # u0 and the first M-1 nodes, extrapolated to the last node
        self.extrap = np.array([np.prod([(c[-1] - x[i]) / (x[j] - x[i]) for i in range(M) if i != j]) for j in range(M)])
        self.f_end = None
        self.active_history = []  # (time, active fraction) per sweep

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _c(cells):
        return slice(None) if cells is None else cells

    def _neighbours(self, cells):
        """Indices of ``cells`` plus their Laplacian-stencil neighbours."""
        P = self.level.prob
        grid = np.zeros(self.n, bool)
        grid[cells] = True
        grid = grid.reshape(P.parabolic.shape)
        cross = np.zeros((3,) * grid.ndim, bool)
        for d in range(grid.ndim):
            idx = [1] * grid.ndim
            for s in (0, 1, 2):
                idx[d] = s
                cross[tuple(idx)] = True
        return np.flatnonzero(binary_dilation(grid, structure=cross))

    def _ensure_phi(self, cells):
        """Rates lambda(u0), phi functions and exponential quadrature weights for the cells that lack them."""
        L, P = self.level, self.level.prob
        need = np.flatnonzero(~self.phi_ready) if cells is None else cells[~self.phi_ready[cells]]
        if need.size == 0:
            return
        M, c = self.coll.num_nodes, self.coll.nodes
        u0c = np.ascontiguousarray(np.asarray(L.u[0])[:, need])
        lm = np.zeros_like(u0c)
        P.im_lmbda_exp(u0c, lm)
        P.work_counters['lmbda'].niter += need.size / self.n
        lm = lm[P.rhs_exp_indeces]
        phi = np.empty((M, M) + lm.shape)
        self.phi_eval(L.dt * c, list(range(1, M + 1)), phi, lm)
        k = np.arange(0, M)
        wgt = self.w[None, :, :] * c[:, None, None] ** (k[None, None, :] + 1)
        self.lmbda[:, need] = lm
        self.Qmat_exp[..., need] = np.sum(wgt[:, :, :, None, None] * phi[:, None, :, :, :], axis=2)
        self.phi_ready[need] = True

    def _eval_f_cells(self, m, cells):
        """Re-evaluate f at node m: the ionic model on ``cells`` only, diffusion everywhere."""
        L, P = self.level, self.level.prob
        t = L.time + L.dt * self.coll.nodes[m - 1]
        f = L.f[m]
        if cells is None:
            P.eval_f(L.u[m], t, fh=f)
            return
        u = np.asarray(L.u[m])
        uc = np.ascontiguousarray(u[:, cells])
        fe = np.zeros_like(uc)
        P.im_f_nonstiff(uc, fe)  # writes only its own indices, the rest stay zero
        stim = P.Istim(t)  # a scalar 0 when no stimulus is on
        fe[0] += stim[cells] if np.ndim(stim) else stim
        lm, yi = np.zeros_like(uc), np.zeros_like(uc)
        P.im_lmbda_yinf_exp(uc, lm, yi)
        fx = np.zeros_like(uc)
        ex = P.im_exp_indeces
        fx[ex] = lm[ex] * (uc[ex] - yi[ex])
        f.expl[:, cells] = fe
        f.exp[:, cells] = fx
        f.impl[0][:] = 0.0
        P.parabolic.add_disc_laplacian(u[0], f.impl[0])
        frac = cells.size / self.n
        P.work_counters['f_expl'].niter += frac
        P.work_counters['lmbda_yinf'].niter += frac
        P.work_counters['laplacian']()

    # ------------------------------------------------------------------ sweeper interface
    def predict(self):
        L, P = self.level, self.level.prob
        M = self.coll.num_nodes
        if self.f_end is not None and abs(self.f_end[0] - L.time) < 1e-9 * L.dt:
            L.f[0] = P.dtype_f(self.f_end[1])  # f(u0) is f at the previous step's last node
        else:
            L.f[0] = P.eval_f(L.u[0], L.time)
        for m in range(1, M + 1):
            L.u[m] = P.dtype_u(L.u[0])
            L.f[m] = P.dtype_f(L.f[0])

        u0 = np.asarray(L.u[0])
        ns, self.n = u0.shape
        ne = len(P.rhs_exp_indeces)
        self.u0norm = np.linalg.norm(u0, axis=1)
        scale = self.u0norm / np.sqrt(self.n)  # RMS per state
        self.res = _Tracked(M, ns, self.n, scale)
        self.inc = _Tracked(1, ns, self.n, scale) if self.params.mask == 'inc' else None
        if not hasattr(self, 'Qmat_exp') or self.Qmat_exp.shape[-1] != self.n:
            self.Qmat_exp = np.zeros((M, M, ne, self.n))
            self.lmbda = np.zeros((ne, self.n))
        self.phi_ready = np.zeros(self.n, bool)
        self.first = True
        self.k = 0
        self.ecell = np.zeros(self.n)
        self.scale = scale

        if self.params.mask == 'none':
            self.dirty = None
        else:
            ft = np.asarray(L.f[0].impl) + np.asarray(L.f[0].expl) + np.asarray(L.f[0].exp)
            bound = L.dt * self.coll.nodes[:, None, None] * ft[None]  # exact residual, or a bound for the gates
            quiet = (np.abs(bound) / scale[None, :, None]).max(axis=(0, 1)) <= self.params.theta * L.params.restol
            self.res.set(None, np.where(quiet[None, None, :], bound, 0.0))
            self.dirty = np.flatnonzero(~quiet)
        self.active = None
        L.status.unlocked = True
        L.status.updated = True

    def compute_lambda_phi_Qmat_exp(self):
        pass  # done lazily per cell by _ensure_phi

    def integrate_cells(self, cells):
        """Exponential collocation integrals I_m (from the step start) on ``cells``, shape (M, nstates, ncells)."""
        L, P = self.level, self.level.prob
        M = self.coll.num_nodes
        self._ensure_phi(cells)
        c = self._c(cells)
        st, ns, ex = P.rhs_stiff_indeces, P.rhs_nonstiff_indeces, P.rhs_exp_indeces
        Qm = self.coll.Qmat[1:, 1:]
        u0 = np.asarray(L.u[0])
        nc = self.n if cells is None else cells.size
        me = np.zeros((M, u0.shape[0], nc))
        me[:, st] += np.einsum('mj,jsd->msd', Qm, np.stack([np.asarray(L.f[j].impl)[st][:, c] for j in range(1, M + 1)]))
        me[:, ns] += np.einsum('mj,jsd->msd', Qm, np.stack([np.asarray(L.f[j].expl)[ns][:, c] for j in range(1, M + 1)]))
        lm, u0e = self.lmbda[:, c], u0[ex][:, c]
        Qe = np.stack([np.asarray(L.f[j].exp)[ex][:, c] + lm * (u0e - np.asarray(L.u[j])[ex][:, c]) for j in range(1, M + 1)])
        me[:, ex] += np.einsum('mjed,jed->med', self.Qmat_exp[..., c], Qe)
        me *= L.dt
        return me

    def integrate(self):
        me = self.integrate_cells(None)
        return [self.level.prob.dtype_u(me[m]) for m in range(self.coll.num_nodes)]

    def compute_residual(self, stage=''):
        L = self.level
        M = self.coll.num_nodes
        assert all(t is None for t in L.tau), 'single level only'
        d = self.dirty
        if d is None or d.size:
            c = self._c(d)
            U = np.stack([np.asarray(L.u[m])[:, c] for m in range(1, M + 1)])
            self.res.set(d, np.asarray(L.u[0])[:, c][None] + self.integrate_cells(d) - U)
        if self.params.mask == 'err':
            thr = np.full(self.n, self.params.theta * L.params.restol)
            if d is None or d.size:
                c = self._c(d)
                U = [np.asarray(L.u[0])[:, c]] + [np.asarray(L.u[m])[:, c] for m in range(1, M)]
                e = np.abs(np.asarray(L.u[M])[:, c] - sum(w * u for w, u in zip(self.extrap, U)))
                self.ecell[c] = (e / self.scale[:, None]).max(axis=0)
            if self.k >= self.params.kmin:
                thr = np.maximum(thr, self.params.kappa * self.ecell)
            ratio = self.res.cellmax / thr
            self.active = np.flatnonzero(ratio > 1.0)
            # below restol exactly when every cell is below its own tolerance
            L.status.residual = L.params.restol * min(ratio.max(), 1e3) if self.active.size else 0.0
            L.status.updated = False
            return
        tracked = self.inc if (self.params.mask == 'inc' and not self.first) else self.res
        L.status.residual = tracked.rel_norm(self.u0norm)
        if L.status.residual < L.params.restol:  # confirm without accumulated round-off
            L.status.residual = tracked.rel_norm(self.u0norm, exact=True)
        if self.params.mask == 'res' or (self.params.mask == 'inc' and self.first):
            self.active = np.flatnonzero(self.res.cellmax > self.params.theta * L.params.restol)
        L.status.updated = False

    def update_nodes(self):
        L, P = self.level, self.level.prob
        assert L.status.unlocked
        M = self.coll.num_nodes
        cells = self.active
        c = self._c(cells)
        self.active_history.append((L.time, 1.0 if cells is None else cells.size / self.n))

        keep = None  # 'inc' keeps no residual outside the active cells, so only theirs drives the diffusion solve
        if cells is not None and self.params.mask == 'inc':
            keep = np.zeros(self.n, bool)
            keep[cells] = True
        incmax = None
        for m in range(1, M + 1):
            r = self.res.A[m - 1]
            rV = r[0] if keep is None else np.where(keep, r[0], 0.0)
            dV = np.empty(self.n)
            P.work_counters['solve']()
            P.parabolic.solve_system(rV, L.dt * self.q[m - 1], None, L.time, dV)
            d = r[:, c].copy()
            d[0] = dV[c]
            np.asarray(L.u[m])[:, c] += d
            incmax = np.abs(d) if incmax is None else np.maximum(incmax, np.abs(d))
            self._eval_f_cells(m, cells)

        if self.params.mask in ['res', 'err']:
            self.dirty = None if cells is None else self._neighbours(cells)
        elif self.params.mask == 'inc':
            self.inc.set(cells, incmax[None])
            hot = np.flatnonzero(self.inc.cellmax > self.params.theta * L.params.restol)
            self.active = self._neighbours(hot)
            self.dirty = self.active
        self.first = False
        self.k += 1
        L.status.updated = True

    def compute_end_point(self):
        super().compute_end_point()
        L = self.level
        self.f_end = (L.time + L.dt, self.level.prob.dtype_f(L.f[-1]))
