from dedalus import public as de
from dedalus import core
import numpy as np
import matplotlib.pyplot as plt

import time

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field


class wrapper(core.field.Field):
    def __abs__(self):

        abs = super(wrapper, self).__abs__()
        while hasattr(abs, 'evaluate'):
            abs = abs.evaluate()
        print(type(abs), self)
        return np.amax(abs['g'])


mytype = core.field.Field

de.logging_setup.rootlogger.setLevel('INFO')

xbasis = de.Fourier('x', 1024, interval=(0, 1), dealias=1)

domain = de.Domain([xbasis], grid_dtype=np.float64, comm=None)

print(domain.global_grid_shape(), domain.local_grid_shape())

x = domain.grid(0, scales=1)

d1 = dedalus_field(domain)
d1.values['g'] = np.sin(2 * np.pi * x)
d2 = dedalus_field(domain)
d2.values['g'] = np.sin(2 * np.pi * x)

print((d1 + d2).values['g'])


d1 = mytype(domain)
d1['g'] = np.sin(2 * np.pi * x)
d2 = mytype(domain)
d2['g'] = np.sin(2 * np.pi * x)

print((d1 + d2).evaluate()['g'])
print((d1 - d2).evaluate()['g'])
print((2.0 * d2).evaluate()['g'])
print(np.amax(abs(d2).evaluate()['g']))


d1 = wrapper(domain)
d1['g'] = np.sin(2 * np.pi * x)
d2 = wrapper(domain)
d2['g'] = np.sin(2 * np.pi * x)

print((d1 + d2).evaluate()['g'])
print((d1 - d2).evaluate()['g'])
print((2.0 * d2).evaluate()['g'])
print(abs(d2))
print(np.amax(abs(d1 + d2).evaluate()['g']))
# print(np.amax(abs(d2).evaluate()['g']))
exit()


g = domain.new_field()
g['g'] = np.sin(2 * np.pi * x)

t0 = time.perf_counter()
for i in range(10000):
    f = domain.new_field()
    f['g'] = g['g'] + g['g']
    # f['c'][:] = g['c'][:]
t1 = time.perf_counter()
print(t1 - t0)

t0 = time.perf_counter()
for i in range(10000):
    f = (g + g).evaluate()
    # f['c'][:] = g['c'][:]
t1 = time.perf_counter()
print(t1 - t0)

t0 = time.perf_counter()
for i in range(10000):
    f = np.zeros(tuple(domain.global_grid_shape()))
    f = g['g'] + g['g']
    # f['c'] = g['c']
t1 = time.perf_counter()
print(t1 - t0)

t0 = time.perf_counter()
for i in range(10000):
    # f = wrapper(domain)
    f = g + g
    # f['c'] = g['c']
t1 = time.perf_counter()
print(t1 - t0)


# fxx = xbasis.Differentiate(f)
#
# g = de.operators.FieldCopyField(f).evaluate()
#
# print((f + g).evaluate())
#
#
# f['g'][:] = 1.0
# print(f['g'], g['g'])
# print(f, g)
