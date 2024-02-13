import numpy as np
import scipy as sp

L = 10.0
p = 6
Nf = 2**p
Nc = Nf // 2

xf = np.linspace(0, L, 2 * Nf + 1)
xf = xf[1::2]
dxf = xf[1] - xf[0]
xc = np.linspace(0, L, 2 * Nc + 1)
xc = xc[1::2]

norm = "forward"


def f(x):
    return x < L / 2.0
    return np.cos(2 * np.pi * x / L)  # * np.cos(2 * np.pi * 4 * x / L) * np.cos(2 * np.pi * x * 10 / L) * x**2 * (L - x) ** 2


def ddf(x):
    return np.zeros_like(x)
    return -((2 * np.pi / L) ** 2) * np.cos(2 * np.pi * x / L)


fxf = f(xf)
fxc = f(xc)


# Send fine sol to coarse grid by truncating higher modes
fxc_dct = sp.fft.idct(sp.fft.dct(fxf, norm=norm), n=Nc, norm=norm)
err_c = fxc_dct - fxc
err_c = np.linalg.norm(err_c, np.inf)
print(f"Err on fc: {err_c}")

# Extend coarse sol to fine grid by setting higher modes to zero
fxf_dct = sp.fft.idct(sp.fft.dct(fxc, norm=norm), n=Nf, norm=norm)
err = fxf_dct - fxf
err = np.linalg.norm(err, np.inf)
print(f"Err dct on ff: {err}")

# Solve Laplace equation
ddf_hat = sp.fft.dct(ddf(xf), norm=norm)
lap_dct = (2 * np.cos(np.pi * np.arange(Nf) / Nf) - 2.0) / dxf**2
f_hat = ddf_hat / lap_dct
f_hat[0] = 0
sol_laplace = sp.fft.idct(f_hat, norm=norm)
err = sol_laplace - fxf
err = np.linalg.norm(err, np.inf)
print(f"Err Laplace equation: {err}")

# Compute Laplacian in frequency domain
fxf_hat = sp.fft.dct(fxf, norm=norm)
ddfx_hat = lap_dct * fxf_hat
ddfx = sp.fft.idct(ddfx_hat, norm=norm)
err = ddfx - ddf(xf)
err = np.linalg.norm(err, np.inf)
print(f"Err on Laplacian: {err}")

# plot of fx
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(xf, fxf, color='k', marker=None, label="f(x)")
ax1.plot(xc, fxc, color='blue', marker=None, label="f new")
ax2.plot(np.arange(0, fxf_hat.size), fxf_hat)
# ax2.plot(np.arange(0, fxc_hat.size), fxc_hat)
plt.show()
