from dedalus import public as de
from dedalus import core
import numpy as np
import matplotlib.pyplot as plt

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field

# class wrapper(core.field.Field):
#
#     # def __init__(self, domain):
#     #     super(wrapper, self).__init__(domain)
#
#     def __add__(self, other):
#
#         return wrapper(super(wrapper, self).__add__(other).evaluate(), self.domain)
#     #
#     def __abs__(self):
#         return abs(self).evaluate()


de.logging_setup.rootlogger.setLevel('INFO')

xbasis = de.Fourier('x', 16, interval=(0, 1), dealias=1)


domain = de.Domain([xbasis], grid_dtype=np.float64, comm=None)
domain_2 = de.Domain([xbasis], grid_dtype=np.float64, comm=None)

print(domain.global_grid_shape(), domain.local_grid_shape())

f = domain.new_field()
fxx = xbasis.Differentiate(f)

g = de.operators.FieldCopyField(f).evaluate()

print((f + g).evaluate())

f['g'][:] = 1.0
print(f['g'], g['g'])
print(f, g)


# exit()
g = domain.new_field()

f2 = domain_2.new_field()

try:
    print(f + f2)
except ValueError:
    print('Non-unique domains')

x = domain.grid(0, scales=1)

f['g'] = np.sin(2 * np.pi * x)
print(fxx.evaluate()['g'])
f['g'] = np.cos(2 * np.pi * x)
print(fxx.evaluate()['g'])
exit()


g['g'] = np.cos(2 * np.pi * x)

u = domain.new_field()

# xbasis_c = de.Fourier('x', 4, interval=(0,1), dealias=1)
# domain_c = de.Domain([xbasis_c],grid_dtype=np.float64, comm=None)
#
# fex = domain.new_field()
# fex['g'] = np.copy(f['g'])
#
# ff = domain.new_field()
# ff['g'] = np.copy(f['g'])
# ff.set_scales(scales=0.5)
#
# fc = domain_c.new_field()
# fc['g'] = ff['g']
#
#
# print(fc['g'].shape, fex['g'].shape)
#
# print((fc-fex).evaluate()['g'])

# exit()

h = (f + g).evaluate()


hxx = de.operators.differentiate(h, x=2).evaluate()

hxxex = domain.new_field()
hxxex['g'] = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x) - (2 * np.pi) ** 2 * np.cos(2 * np.pi * x)

print(max(abs(hxx - hxxex).evaluate()['g']))
# exit()

forcing = domain.new_field()
forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(0) - (np.pi * 2) ** 2 * np.cos(0))

dt = 0.1 / 16

u_old = domain.new_field()
u_old['g'] = np.copy(f['g'])

problem = de.LinearBoundaryValueProblem(domain=domain, variables=['u'])
problem.meta[:]['x']['dirichlet'] = True
problem.parameters['dt'] = dt
problem.parameters['u_old'] = u_old + dt * forcing
problem.add_equation("u - dt * dx(dx(u)) = u_old")


solver = problem.build_solver()
u = solver.state['u']

Tend = 1.0
nsteps = int(Tend / dt)

t = 0.0
for n in range(nsteps):
    problem.parameters['u_old'] = u_old + dt * forcing
    solver.solve()
    t += dt
    forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(t) - (np.pi * 2) ** 2 * np.cos(t))
    u_old['g'] = np.copy(u['g'])
    # print(n)


uex = domain.new_field()
# uex['g'] = np.sin(2*np.pi*x) * np.exp(-(2*np.pi)**2 * Tend)
uex['g'] = np.sin(2 * np.pi * x) * np.cos(Tend)

print(np.linalg.norm(u['g'] - uex['g'], np.inf))

# plt.figure(1)
# plt.plot(x,u['g'])
# plt.plot(x,uex['g'])
#
# plt.pause(1)
#
# exit()

# forcing = domain.new_field()
# forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(0) - (np.pi * 2) ** 2 * np.cos(0))

# u_old['g'] = np.zeros(domain.global_grid_shape())
problem = de.IVP(domain=domain, variables=['u'])
# problem.parameters['RHS'] = u_old + forcing
problem.parameters['RHS'] = 0
problem.add_equation("dt(u) - dx(dx(u)) = RHS")

ts = de.timesteppers.SBDF1
solver = problem.build_solver(ts)
u = solver.state['u']
# u['g'] = np.sin(2*np.pi*x)
tmp = np.tanh((0.25 - np.sqrt((x - 0.5) ** 2)) / (np.sqrt(2) * 0.04))
u['g'] = tmp

dt = 1.912834231231e07
t = 0.0
for n in range(nsteps):
    # u['g'] = u['g'] - dt * np.sin(np.pi * 2 * x) * (np.sin(t) - (np.pi * 2) ** 2 * np.cos(t))
    solver.step(dt)
    t += dt

    uxx = de.operators.differentiate(u, x=2).evaluate()
    print(max(abs(u['g'] - dt * uxx['g'] - tmp)))
    exit()

print(np.linalg.norm(u['g'] - uex['g'], np.inf))

# #
# plt.figure(1)
# plt.plot(x,u['g'])
# plt.plot(x,uex['g'])
# #
# plt.pause(1)
