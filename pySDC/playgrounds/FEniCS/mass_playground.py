import dolfin as df
import numpy as np
import matplotlib.pyplot as plt


N = 128  # Number of elements
k = 127  # Wave frequency
d = 1  # FE order

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

# Plot
plt.figure()
plt.plot(abs(fw2))

plt.figure()
plt.plot(abs(fMw2))

plt.show()

