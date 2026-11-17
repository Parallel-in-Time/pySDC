import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

import pySDC.helpers.plot_helper as plt_helper

import pylustrator

# pylustrator.start()


N = 128  # Number of elements
k = 7  # Wave frequency
d = 4  # FE order

# Get mesh and function space (CG or DG)
mesh = df.UnitIntervalMesh(N)
V = df.FunctionSpace(mesh, "CG", d)
# V = df.FunctionSpace(mesh, "DG", d)

# Build mass matrix
u = df.TrialFunction(V)
v = df.TestFunction(V)
a_M = u * v * df.dx
M = df.assemble(a_M)

# Create vector with sine function
u0 = df.Expression('sin(k*pi*x[0])', pi=np.pi, k=k, degree=d)
w = df.interpolate(u0, V)

# Apply mass matrix to this vector
Mw = df.Function(V)
M.mult(w.vector(), Mw.vector())

# Do FFT to get the frequencies
fw = np.fft.fft(w.vector()[:])
fMw = np.fft.fft(Mw.vector()[:])
# Shift to have zero frequency in the middle of the plot
fw2 = np.fft.fftshift(fw)
fMw2 = np.fft.fftshift(fMw)

fw2 /= np.amax(abs(fw2))
fMw2 /= np.amax(abs(fMw2))

ndofs = fw.shape[0]

# Plot
plt_helper.setup_mpl()

plt_helper.newfig(240, 1, ratio=0.8)
plt_helper.plt.plot(abs(fw2), lw=2, label=f'N = {N} \n degree = {d} \n wave number = {k}')
plt_helper.plt.xticks(
    [0, ndofs / 4 - 1, ndofs / 2 - 1, 3 * ndofs / 4 - 1, ndofs - 1],
    (r'-$\pi$', r'-$\pi/2$', r'$0$', r'+$\pi$/2', r'+$\pi$'),
)
plt_helper.plt.xlabel('spectrum')
plt_helper.plt.ylabel('normed amplitude')
# plt_helper.plt.legend()
plt_helper.plt.grid()
plt_helper.savefig('spectrum_noM_CG')
# plt_helper.savefig('spectrum_noM_DG')
# plt_helper.savefig('spectrum_noM_DG_8')

plt_helper.newfig(240, 1, ratio=0.8)
plt_helper.plt.plot(abs(fMw2), lw=2, label=f'N = {N} \n degree = {d} \n wave number = {k}')
plt_helper.plt.xticks(
    [0, ndofs / 4 - 1, ndofs / 2 - 1, 3 * ndofs / 4 - 1, ndofs - 1],
    (r'-$\pi$', r'-$\pi/2$', r'$0$', r'+$\pi$/2', r'+$\pi$'),
)
plt_helper.plt.xlabel('spectrum')
plt_helper.plt.ylabel('normed amplitude')
# plt_helper.plt.legend()
plt_helper.plt.grid()
plt_helper.savefig('spectrum_M_CG')
# plt_helper.savefig('spectrum_M_DG')
# plt_helper.savefig('spectrum_M_DG_8')

# plt.show()
