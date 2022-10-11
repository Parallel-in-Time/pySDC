######################################################################################
# Roberts advection test for IMEX sdc in Paralpha with just an explicit part and Euler
# no spatial variables here, just time
######################################################################################

import numpy as np


# Paralpha settings
N = int(1e03)  # spatial points
dt = 1e-03
Tend = 1e-02
L = int(Tend / dt)

alpha = 1e-01
K = 5  # maxiter

# equation settings
T1 = 0
T2 = L * dt + T1
X1 = 0
X2 = 1
c = 1
x = np.linspace(X1, X2, N + 2)[1:-1]
dx = x[1] - x[0]
t = np.linspace(T1, T2, L + 1)[1:]

print(f'CFL: {c*dt/dx}')

print('solving on [{}, {}] x [{}, {}]'.format(T1, T2, X1, X2))

# functions
A = 1 / (1 * dx) * (-np.eye(k=-1, N=N) + np.eye(k=0, N=N))
A[0, -1] = -1 / (1 * dx)
# A = 1/(2 * dx) * (-np.eye(k=-1, N=N) + np.eye(k=1, N=N))
# A[0, -1] = -1/(2 * dx)
# A[-1, 0] = 1/(2 * dx)


def u_exact(t, x):
    return np.sin(2 * np.pi * (x - c * t))


def f_exp(t, x, u):
    y = -c * A @ u
    return y


# the rest
u0 = u_exact(T1, x)

# explicit euler
us = u0.copy()
for l in range(L):
    us = us + dt * f_exp(t[l] - dt, x, us)

err_euler = np.linalg.norm((us - u_exact(T2, x)).flatten(), np.inf)
print('seq err = ', err_euler)

Ea = np.eye(k=-1, N=L)  # + alpha * np.eye(k=-1, N=L)
Ea[0, -1] = alpha

print(sum([np.linalg.matrix_power(Ea, l) for l in range(10)]))

print(np.linalg.norm(np.linalg.inv(Ea), np.inf))
exit()

print(np.linalg.norm(np.eye(k=-1, N=L) + Ea, np.inf))


u = np.zeros(N * L, dtype=complex)

# for l in range(L):
#     u[l * N: (l + 1) * N] = u0

u[:N] = u0

rhs = np.empty(N * L, dtype=complex)

d, S = np.linalg.eig(Ea)
Sinv = np.linalg.inv(S)  # S @ d @ Sinv = Ea

print(f'Diagonalization error: {np.linalg.norm(S @ np.diag(d) @ Sinv - Ea, np.inf)}')
# exit()
err = np.linalg.norm(u[-N:] - u_exact(T2, x), np.inf)
print(err, 0)

for k in range(K):
    rhs[:N] = u0 - alpha * u[-N:] + dt * f_exp(t[0] - dt, x, u[:N])
    for l in range(1, L, 1):
        rhs[l * N : (l + 1) * N] = dt * f_exp(t[l] - dt, x, u[(l - 1) * N : l * N]) - alpha * u[(l - 1) * N : l * N]

    # u = np.kron(Sinv, np.eye(N)) @ rhs
    for i in range(L):
        temp = np.zeros(N, dtype=complex)
        for j in range(L):
            temp += Sinv[i, j] * rhs[j * N : (j + 1) * N]
        u[i * N : (i + 1) * N] = temp.copy()

    # solve diagonal systems
    for l in range(L):
        u[l * N : (l + 1) * N] /= 1 + d[l]

    # u = np.kron(S, np.eye(N)) @ u
    u1 = u.copy()
    for i in range(L):
        temp = np.zeros(N, dtype=complex)
        for j in range(L):
            temp += S[i, j] * u1[j * N : (j + 1) * N]
        u[i * N : (i + 1) * N] = temp.copy()

    err_paralpha = np.linalg.norm((u[-N:] - u_exact(T2, x)).flatten(), np.inf)
    print(err_paralpha, k + 1)
    if err_euler > err_paralpha:
        break


err = np.linalg.norm((us - u[-N:]).flatten(), np.inf)
print('error between seq and paralpha = ', err)

# gc. collect()
