r"""
A one-dimensional Kohn-Sham molecule for real-time TDDFT, with the structure of a production pseudopotential code.

Several doubly occupied orbitals :math:`\psi_j` evolve in a local pseudopotential, a nonlocal Kleinman-Bylander
projector per atom, the Hartree and a local exchange potential of the density, a laser in length gauge and a complex
absorbing potential (CAP) at the ends of the box:

.. math::
    i\partial_t\psi_j = -\tfrac12\partial_x^2\psi_j + \big(v_{loc} + v_H[\rho] + v_x[\rho] + xE(t) - i\,v_{cap}\big)
    \psi_j + \sum_a E_{kb}\,|\beta_a\rangle\langle\beta_a|\psi_j\rangle, \qquad \rho = 2\sum_j|\psi_j|^2.

Nucleus and electron interaction are soft-Coulomb. The exchange potential is the 3D LDA formula
:math:`v_x = -(3\rho/\pi)^{1/3}`, used as a model local nonlinearity, not as a 1D functional. Space is
pseudo-spectral (FFT); the kinetic part is the implicit one of the IMEX splitting (an exact FFT solve, as in 3D
plane-wave codes), everything else is explicit.
"""

import numpy as np
import scipy.linalg as sl

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class KohnShamMolecule1D(Problem):
    r"""
    One-dimensional Kohn-Sham molecule driven by a laser pulse, with nonlocal pseudopotentials.

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Length of the periodic box (the CAP keeps the wrap-around away).
    atoms : tuple of float
        Positions of the nuclei.
    Z : float
        Charge of each nucleus (soft-Coulomb with softening ``a_nuc``).
    a_nuc, a_ee : float
        Softening of the nuclear and of the electron-electron interaction.
    E_kb, r_kb : float
        Strength and width of the Gaussian Kleinman-Bylander projector on each atom.
    norb : int
        Number of doubly occupied orbitals.
    E0, omega, ncycles : float, float, int
        Peak field, frequency and number of cycles of the :math:`\sin^2` laser pulse.
    x_cap, eta_cap : float
        Onset and strength of the quadratic complex absorbing potential.
    laser : bool
        Switch the laser off for field-free runs.

    Notes
    -----
    Work counters: ``fft`` counts orbital-sized FFT round trips (an H application or a kinetic solve costs one per
    orbital, the unit of cost here), ``proj`` projector applications (one per orbital, cheap) and ``density``
    density updates (Hartree and exchange, shared by all orbitals).

    After a kinetic solve the implicit right-hand side is known without an FFT, :math:`f_{impl} = (u - r)/\alpha`,
    so ``eval_f`` reuses it when called on the solution just computed.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self,
        N=1024,
        L=204.8,
        atoms=(-2.0, 2.0),
        Z=3.0,
        a_nuc=1.0,
        a_ee=1.0,
        E_kb=1.0,
        r_kb=0.7,
        norb=3,
        E0=0.0534,
        omega=0.057,
        ncycles=4,
        x_cap=70.0,
        eta_cap=2e-3,
        laser=True,
    ):
        super().__init__(init=((norb, N), None, np.dtype('complex128')))
        self._makeAttributeAndRegister(
            'N',
            'L',
            'atoms',
            'Z',
            'a_nuc',
            'a_ee',
            'E_kb',
            'r_kb',
            'norb',
            'E0',
            'omega',
            'ncycles',
            'x_cap',
            'eta_cap',
            'laser',
            localVars=locals(),
            readOnly=True,
        )
        self.dx = L / N
        self.x = (np.arange(N) - N // 2) * self.dx
        self.T = 0.5 * (2 * np.pi * np.fft.fftfreq(N, self.dx)) ** 2
        self.v_loc = sum(-Z / np.sqrt((self.x - X) ** 2 + a_nuc**2) for X in atoms)
        self.v_cap = np.where(np.abs(self.x) > x_cap, eta_cap * (np.abs(self.x) - x_cap) ** 2, 0.0)
        beta = np.array([np.exp(-((self.x - X) ** 2) / (2 * r_kb**2)) for X in atoms])
        self.beta = beta / np.sqrt(np.sum(beta**2, axis=1, keepdims=True) * self.dx)
        # kernel of the linear (zero-padded) convolution for the Hartree potential
        xp = (np.arange(2 * N) - N) * self.dx
        self.w_hat = np.fft.fft(np.fft.ifftshift(1.0 / np.sqrt(xp**2 + a_ee**2))) * self.dx
        self.T_pulse = ncycles * 2 * np.pi / omega
        self.n_electrons = 2 * norb
        self._solved = None
        for key in ['fft', 'proj', 'density']:
            self.work_counters[key] = WorkCounter()

    # --- the pieces of the Hamiltonian ----------------------------------------------------------------------------
    def E(self, t):
        """Laser field at time t"""
        if not self.laser or t < 0 or t > self.T_pulse:
            return 0.0
        return self.E0 * np.sin(np.pi * t / self.T_pulse) ** 2 * np.sin(self.omega * t)

    def density(self, psi):
        """Electron density of the orbitals (two electrons each)"""
        return 2 * np.sum(np.abs(psi) ** 2, axis=0)

    def v_hx_rho(self, rho):
        """Hartree plus model exchange potential of a density"""
        rp = np.zeros(2 * self.N)
        rp[: self.N] = rho
        vh = np.fft.ifft(np.fft.fft(rp) * self.w_hat).real[: self.N]  # index i is grid point x_i again
        return vh - np.cbrt(3 * rho / np.pi)

    def potential(self, psi, t):
        """Total local potential of the current orbitals (one density update), complex because of the CAP"""
        self.work_counters['density']()
        return self.v_loc + self.v_hx_rho(self.density(psi)) + self.x * self.E(t) - 1j * self.v_cap

    def apply_nonlocal(self, psi):
        r""":math:`\sum_a E_{kb} |\beta_a\rangle\langle\beta_a|\psi_j\rangle` for every orbital"""
        self.work_counters['proj'].niter += psi.shape[0]
        return self.E_kb * (((psi @ self.beta.T) * self.dx) @ self.beta)

    def kinetic(self, psi):
        r""":math:`-\tfrac12\partial_x^2` for every orbital"""
        self.work_counters['fft'].niter += psi.shape[0]
        return np.fft.ifft(self.T * np.fft.fft(psi, axis=-1), axis=-1)

    def apply_H(self, psi, t, V=None):
        """Hamiltonian applied to the orbitals, with the local potential given or computed from them"""
        if V is None:
            V = self.potential(psi, t)
        return self.kinetic(psi) + V * psi + self.apply_nonlocal(psi)

    # --- pySDC interface ------------------------------------------------------------------------------------------
    def eval_f(self, u, t):
        """Right-hand side: kinetic part implicit, potential, projectors and laser explicit"""
        f = self.dtype_f(self.init)
        if self._solved is not None and self._solved[0] is u:
            f.impl[:] = self._solved[1]
        else:
            f.impl[:] = -1j * self.kinetic(u)
        self._solved = None
        f.expl[:] = -1j * (self.potential(u, t) * u + self.apply_nonlocal(u))
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""Solve :math:`u + i\,\alpha\,T u = r` exactly in Fourier space, orbital by orbital"""
        self.work_counters['fft'].niter += rhs.shape[0]
        me = self.dtype_u(self.init)
        me[:] = np.fft.ifft(np.fft.fft(rhs, axis=-1) / (1.0 + 1j * factor * self.T), axis=-1)
        if factor != 0:
            self._solved = (me, (me - rhs) / factor)
        return me

    # --- ground state and observables -----------------------------------------------------------------------------
    def ground_state(self, tol=1e-11, maxiter=400):
        """
        Self-consistent ground state by dense diagonalization and Anderson mixing of the density (setup only, not
        counted). Sets ``eps`` (the lowest orbital energies) and resets the work counters.
        """
        Tmat = np.real(np.fft.ifft(self.T[:, None] * np.fft.fft(np.eye(self.N), axis=0), axis=0))
        Vnl = self.E_kb * self.beta.T @ self.beta * self.dx
        rho_in = self.n_electrons * np.exp(-(self.x**2) / 20) / np.sqrt(20 * np.pi)
        X, F = [], []
        for _ in range(maxiter):
            w, v = sl.eigh(Tmat + np.diag(self.v_loc + self.v_hx_rho(rho_in)) + Vnl, subset_by_index=[0, self.norb + 1])
            psi = (v[:, : self.norb] / np.sqrt(self.dx)).T
            res = self.density(psi) - rho_in
            if np.max(np.abs(res)) < tol:
                break
            X, F = (X + [rho_in.copy()])[-6:], (F + [res])[-6:]
            if len(F) > 1:
                dF = np.diff(np.array(F), axis=0)
                dX = np.diff(np.array(X), axis=0)
                g = np.linalg.lstsq(dF.T, res, rcond=None)[0]
                rho_in = rho_in + 0.3 * res - (dX.T + 0.3 * dF.T) @ g
            else:
                rho_in = rho_in + 0.3 * res
            rho_in = np.maximum(rho_in, 0.0)
        else:
            raise ProblemError(f'ground state not converged in {maxiter} iterations')
        self.eps = w
        for c in self.work_counters.values():
            c.niter = 0
        u = self.dtype_u(self.init)
        u[:] = psi
        return u

    def dipole(self, psi):
        """Dipole moment of the electron density"""
        return np.sum(self.x * self.density(psi)) * self.dx

    def norm(self, psi):
        """Sum of the orbital norms (decreases when the CAP absorbs ionized electrons)"""
        return np.sum(np.abs(psi) ** 2) * self.dx
