from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import figsize_by_journal


P = RayleighBenard(nx=3, nz=5, Dirichlet_recombination=True, left_preconditioner=True)
dt = 1.0


A = P.M + dt * P.L
A_b = P.put_BCs_in_matrix(P.M + dt * P.L)
A_r = P.spectral.put_BCs_in_matrix(A) @ P.Pr
A_l = P.Pl @ P.spectral.put_BCs_in_matrix(A) @ P.Pr

fig, axs = plt.subplots(1, 4, figsize=figsize_by_journal('TUHH_thesis', 1, 0.4), sharex=True, sharey=True)

for M, ax in zip([A, A_b, A_r, A_l], axs):
    ax.imshow((M / abs(M)).real + (M / abs(M)).imag, rasterized=False, cmap='Spectral')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig('plots/RBC_matrix.pdf', bbox_inches='tight', dpi=300)
plt.show()
