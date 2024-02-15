import numpy as np
import scipy as sp

dim = 3
dom_size = (2.0, 3.0, 4.0)
diffusion = (0.1, 0.2, 0.3)
dom_size = dom_size[:dim]
p = 4
Nf = [int(2 ** np.round(np.log2(L * 2**p))) for L in dom_size]
Nc = [N // 2 for N in Nf]
norm = "forward"


def grids(x):
    dim = len(x)
    if dim == 1:
        return (x[0],)
    elif dim == 2:
        return (x[0][None, :], x[1][:, None])
    elif dim == 3:
        return (x[0][None, None, :], x[1][None, :, None], x[2][:, None, None])


def get_x_dx(dom_size, N):
    x = [np.linspace(0, dom_size[i], 2 * N[i] + 1) for i in range(len(N))]
    x = [xi[1::2] for xi in x]
    dx = [xi[1] - xi[0] for xi in x]
    return grids(x), dx


def u(x):
    if dim == 1:
        ux = np.cos(3.0 * np.pi * x[0] / dom_size[0])
        ux -= ux[0]
    if dim == 2:
        ux = np.cos(3.0 * np.pi * x[0] / dom_size[0]) * np.cos(4.0 * np.pi * x[1] / dom_size[1])
        ux -= ux[0, 0]
    if dim == 3:
        ux = np.cos(3.0 * np.pi * x[0] / dom_size[0]) * np.cos(4.0 * np.pi * x[1] / dom_size[1]) * np.cos(5.0 * np.pi * x[2] / dom_size[2])
        ux -= ux[0, 0, 0]
    return ux


def ddu(x):
    if dim == 1:
        ddux = -((3.0 * np.pi / dom_size[0]) ** 2) * np.cos(3.0 * np.pi * x[0] / dom_size[0])
    if dim == 2:
        ddux = -((3.0 * np.pi / dom_size[0]) ** 2 + (4.0 * np.pi / dom_size[1]) ** 2) * np.cos(3.0 * np.pi * x[0] / dom_size[0]) * np.cos(4.0 * np.pi * x[1] / dom_size[1])
    if dim == 3:
        ddux = (
            -((3.0 * np.pi / dom_size[0]) ** 2 + (4.0 * np.pi / dom_size[1]) ** 2 + (5.0 * np.pi / dom_size[2]) ** 2)
            * np.cos(3.0 * np.pi * x[0] / dom_size[0])
            * np.cos(4.0 * np.pi * x[1] / dom_size[1])
            * np.cos(5.0 * np.pi * x[2] / dom_size[2])
        )
    return ddux


def diff_u(x):
    if dim == 1:
        ddux = -diffusion[0] * ((3.0 * np.pi / dom_size[0]) ** 2) * np.cos(3.0 * np.pi * x[0] / dom_size[0])
    if dim == 2:
        ddux = (
            -(diffusion[0] * (3.0 * np.pi / dom_size[0]) ** 2 + diffusion[1] * (4.0 * np.pi / dom_size[1]) ** 2) * np.cos(3.0 * np.pi * x[0] / dom_size[0]) * np.cos(4.0 * np.pi * x[1] / dom_size[1])
        )
    if dim == 3:
        ddux = (
            -(diffusion[0] * (3.0 * np.pi / dom_size[0]) ** 2 + diffusion[1] * (4.0 * np.pi / dom_size[1]) ** 2 + diffusion[2] * (5.0 * np.pi / dom_size[2]) ** 2)
            * np.cos(3.0 * np.pi * x[0] / dom_size[0])
            * np.cos(4.0 * np.pi * x[1] / dom_size[1])
            * np.cos(5.0 * np.pi * x[2] / dom_size[2])
        )
    return ddux


def lap_dct(N, dx):
    dim = len(N)
    lap = (2.0 * np.cos(np.pi * np.arange(N[0]) / N[0]) - 2.0) / dx[0] ** 2
    if dim >= 2:
        lap = lap[None, :] + np.array((2.0 * np.cos(np.pi * np.arange(N[1]) / N[1]) - 2.0) / dx[1] ** 2)[:, None]
    if dim >= 3:
        lap = lap[None, :, :] + np.array((2.0 * np.cos(np.pi * np.arange(N[2]) / N[2]) - 2.0) / dx[2] ** 2)[:, None, None]
    return lap


def diff_dct(N, dx):
    dim = len(N)
    diff = diffusion[0] * (2.0 * np.cos(np.pi * np.arange(N[0]) / N[0]) - 2.0) / dx[0] ** 2
    if dim >= 2:
        diff = diff[None, :] + diffusion[1] * np.array((2.0 * np.cos(np.pi * np.arange(N[1]) / N[1]) - 2.0) / dx[1] ** 2)[:, None]
    if dim >= 3:
        diff = diff[None, :, :] + diffusion[2] * np.array((2.0 * np.cos(np.pi * np.arange(N[2]) / N[2]) - 2.0) / dx[2] ** 2)[:, None, None]
    return diff


xf, dxf = get_x_dx(dom_size, Nf)
xc, dxc = get_x_dx(dom_size, Nc)
uxf = u(xf)
uxc = u(xc)
xf_shape = uxf.shape
xc_shape = uxc.shape
lap_dct_f = lap_dct(Nf, dxf)
diff_dct_f = diff_dct(Nf, dxf)

# check dct-idct
uxf_dct = sp.fft.idctn(sp.fft.dctn(uxf, norm=norm), s=xf_shape, norm=norm)
err_f = uxf_dct - uxf
err_f = np.abs(err_f).max() / np.abs(uxf).max()
print(f"Err on uf: {err_f}")

# Send fine sol to coarse grid by truncating higher modes
uxc_dct = sp.fft.idctn(sp.fft.dctn(uxf, norm=norm), s=xc_shape, norm=norm)
err_c = uxc_dct - uxc
err_c = np.abs(err_c).max() / np.abs(uxc).max()
print(f"Err dct on uc: {err_c}")

# Extend coarse sol to fine grid by setting higher modes to zero
uxf_dct = sp.fft.idctn(sp.fft.dctn(uxc, norm=norm), s=xf_shape, norm=norm)
err = uxf_dct - uxf
err = np.abs(err).max() / np.abs(uxf).max()
print(f"Err dct on uf: {err}")

# Solve Laplace equation
fxf = ddu(xf)
ddu_hat = sp.fft.dctn(fxf, norm=norm)
u_hat = np.zeros_like(ddu_hat)
u_hat.ravel()[1:] = ddu_hat.ravel()[1:] / lap_dct_f.ravel()[1:]
# u_hat.ravel()[0] = 0
sol_laplace = sp.fft.idctn(u_hat, norm=norm)
sol_laplace -= sol_laplace.ravel()[0]
err = sol_laplace - uxf
err = np.abs(err).max() / np.abs(uxf).max()
print(f"Err Laplace equation: {err}")

# Compute Laplacian in frequency domain and check error in frequency domain and physical domains
uxf_hat = sp.fft.dctn(uxf, norm=norm)
ddux_hat = lap_dct_f * uxf_hat
ddux_hat_ex = sp.fft.dctn(ddu(xf), norm=norm)

err_hat = ddux_hat - ddux_hat_ex
err_hat = np.abs(err_hat).max() / np.abs(ddux_hat_ex).max()
print(f"Err on Laplacian hat: {err_hat}")

ddux_dct = sp.fft.idctn(ddux_hat, norm=norm)
err = ddux_dct - ddu(xf)
err = np.abs(err).max() / np.abs(ddu(xf)).max()
print(f"Err on Laplacian: {err}")

# Solve diffusion equation
fxf = diff_u(xf)
diffu_hat = sp.fft.dctn(fxf, norm=norm)
u_hat = np.zeros_like(diffu_hat)
u_hat.ravel()[1:] = diffu_hat.ravel()[1:] / diff_dct_f.ravel()[1:]
sol_diff = sp.fft.idctn(u_hat, norm=norm)
sol_diff -= sol_diff.ravel()[0]
err = sol_diff - uxf
err = np.abs(err).max() / np.abs(uxf).max()
print(f"Err Diffusion equation: {err}")

# Compute diffusion in frequency domain and check error in frequency domain and physical domains
uxf_hat = sp.fft.dctn(uxf, norm=norm)
diffux_hat = diff_dct_f * uxf_hat
diffux_hat_ex = sp.fft.dctn(diff_u(xf), norm=norm)

err_hat = diffux_hat - diffux_hat_ex
err_hat = np.abs(err_hat).max() / np.abs(diffux_hat_ex).max()
print(f"Err on Diffusion hat: {err_hat}")

diffux_dct = sp.fft.idctn(diffux_hat, norm=norm)
err = diffux_dct - diff_u(xf)
err = np.abs(err).max() / np.abs(diff_u(xf)).max()
print(f"Err on Diffusion: {err}")

# plot of fx
# import matplotlib.pyplot as plt

# if dim == 1:
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.plot(xf[0], uxf, color='k', marker=None, label="f(x)")
#     ax1.plot(xc[0], uxc, color='blue', marker=None, label="f new")
#     ax2.plot(np.arange(0, uxf_hat.size), uxf_hat)
#     # ax2.plot(np.arange(0, fxc_hat.size), fxc_hat)
#     plt.show()
# if dim == 2:
#     from matplotlib import cm

#     fig, (ax1, ax2) = plt.subplots(2, subplot_kw={"projection": "3d"})
#     ax1.plot_surface(xf[0], xf[1], ddux_dct - ddu(xf), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax2.plot_surface(list(np.arange(Nf[0])), list(np.arange(Nf[1])), ddux_hat - sp.fft.dctn(ddu(xf), norm=norm), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     plt.show()
