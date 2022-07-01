import struct
import numpy as np
import types

from pySDC.core.Hooks import hooks
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.datatype_classes.mesh import imex_mesh

from pySDC.projects.Resilience.fault import Fault


class FaultInjector(hooks):

    def __init__(self):
        super(FaultInjector, self).__init__()
        self.fault_frequency_time = np.inf
        self.fault_frequency_iter = np.inf
        self.faults = []
        self.fault_init = []
        self.rnd_params = {}
        self.random_generator = np.random.RandomState(2187)  # number of the cell in which Princess Leia is held

    def add_random_fault(self, time=None, timestep=None, args=types.MappingProxyType({}),
                         rnd_args=types.MappingProxyType({})):
        args['time'] = args.get('time', time)
        args['timestep'] = args.get('timestep', timestep)
        if self.rnd_params == {}:
            self.fault_init += [{'args': args, 'rnd_args': rnd_args}]
        else:
            self.faults += [Fault.random(args=args, rnd_params={**self.rnd_params, **rnd_args},
                            random_generator=self.random_generator)]

    def inject_fault(self, step, f):
        self.logger.info(f'Flipping bit {f.bit} {f.when} iteration {f.iteration} in node {f.node}')
        if f.target == 0:
            if type(step.levels[f.level_number].f[f.node]) == mesh:
                step.levels[f.level_number].f[f.node][f.problem_pos] =\
                    self.flip_bit(step.levels[f.level_number].f[f.node][f.problem_pos][0], f.bit)
            elif type(step.levels[f.level_number].f[f.node]) == imex_mesh:
                step.levels[f.level_number].f[f.node].impl[f.problem_pos] =\
                    self.flip_bit(step.levels[f.level_number].f[f.node].impl[f.problem_pos][0], f.bit)
            else:
                raise NotImplementedError(f'Can\'t flip bits in rhs of type \
{type(step.levels[f.level_number].f[f.node])}')

        elif f.target == 1:
            step.levels[f.level_number].u[f.node][f.problem_pos] =\
                self.flip_bit(step.levels[f.level_number].u[f.node][f.problem_pos][0], f.bit)
        else:
            raise NotImplementedError(f'Target {f.target} for faults not impelemted! Choose 0 for f or 1 for u')

        L = step.levels[f.level_number]
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='bitflip',
                          value=(f.level_number, f.iteration, f.node, f.problem_pos, f.bit, f.target))

        f.happened = True
        self.faults.remove(f)

    def pre_run(self, step, level_number):
        '''
        Store useful quantities for generating random faults here
        '''

        super(FaultInjector, self).pre_run(step, level_number)

        if not type(step.levels[level_number].u[0]) == mesh:
            raise NotImplementedError(f'Fault insertion is only implemented for type mesh, not \
{type(step.levels[level_number].u[0])}')

        self.rnd_params = {
            'level_number': len(step.levels),
            'node': step.levels[0].sweep.params.num_nodes,
            'iteration': step.params.maxiter,
            'problem_pos': step.levels[level_number].u[0].shape,
            'bit': 64,  # change manually if you ever have something else
        }

        for f in self.fault_init:
            self.add_random_fault(args=f['args'], rnd_args=f['rnd_args'])

        if self.rnd_params['level_number'] > 1:
            raise NotImplementedError('I don\'t know how to insert faults in this multi-level madness :(')

        self.timestep_idx = 0
        self.iter_idx = 0

    def pre_step(self, step, level_number):
        super(FaultInjector, self).pre_step(step, level_number)

        self.timestep_idx += 1

        if self.timestep_idx % self.fault_frequency_time == 0 and not self.timestep_idx == 0:
            self.add_random_fault(time=None, timestep=self.timestep_idx)

    def pre_iteration(self, step, level_number):
        '''
        Check if we want to flip a bit here
        '''
        super(FaultInjector, self).pre_iteration(step, level_number)

        if self.iter_idx % self.fault_frequency_iter == 0 and not self.iter_idx == 0:
            self.add_random_fault(time=None, timestep=self.timestep_idx, args={'iteration': step.status.iter})

        # loop though all unhappened faults and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'before']:
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration:
                    self.inject_fault(step, f)

        self.iter_idx += 1

    def post_iteration(self, step, level_number):
        '''
        Check if we want to flip a bit here
        '''
        super(FaultInjector, self).post_iteration(step, level_number)

        # loop though all unhappened faults and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'after']:
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration:
                    self.inject_fault(step, f)

    def to_binary(self, f):
        '''
        Converts a single float in a string containing its binary representation in memory following IEEE754
        The struct.pack function returns the input with the applied conversion code in 8 bit blocks, which are then
        concatenated as a string
        '''
        if type(f) in [np.float64, float]:
            conversion_code = '>d'  # big endian, double
        elif type(f) in [np.float32]:
            conversion_code = '>f'  # big endian, float
        else:
            raise NotImplementedError(f'Don\'t know how to convert number of type {type(f)} to binary')

        return ''.join('{:0>8b}'.format(c) for c in struct.pack(conversion_code, f))

    def to_float(self, s):
        '''
        Converts a string of a IEEE754 binary representation in a float. The string is converted to integer with base 2
        and converted to bytes, which can be unpacked into a Python float by the struct module
        '''
        if len(s) == 64:
            conversion_code = '>d'  # big endian, double
            byte_count = 8
        elif len(s) == 32:
            conversion_code = '>f'  # big endian, float
            byte_count = 4
        else:
            raise NotImplementedError(f'Don\'t know how to convert string of length {len(s)} to float')

        return struct.unpack(conversion_code, int(s, 2).to_bytes(byte_count, 'big'))[0]

    def flip_bit(self, target, bit):
        '''
        Flips a bit at position bit in a target using the bitwise xor operator
        '''
        binary = self.to_binary(target)
        return self.to_float(f'{binary[:bit]}{int(binary[bit]) ^ 1}{binary[bit+1:]}')
