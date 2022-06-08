import struct
import numpy as np

from pySDC.core.Hooks import hooks
from pySDC.implementations.datatype_classes.mesh import mesh


class FaultInjector(hooks):

    def __init__(self):
        super(FaultInjector, self).__init__()
        self.fault_frequency_time = np.inf
        self.fault_frequency_iter = np.inf
        self.random_generator = np.random.RandomState(0)

    def generate_random_fault(self):
        level = self.random_generator.randint(low=0, high=self.num_levels)
        node = self.random_generator.randint(low=0, high=self.num_nodes + 1)
        iteration = self.random_generator.randint(low=0, high=self.maxiter)
        problem_pos = [self.random_generator.randint(low=0, high=i) for i in self.u_shape]
        bit = self.random_generator.randint(low=0, high=self.u_bit_length)
        return level, iteration, node, problem_pos, bit

    def inject_fault(self, step, level_number, level, iteration, node, problem_pos, bit):
        step.levels[level_number].u[node][problem_pos] =\
            self.flip_bit(step.levels[level_number].u[node][problem_pos][0], bit)

        L = step.levels[level_number]
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='bitflip', value=(level, iteration, node, problem_pos, bit))

    def pre_run(self, step, level_number):
        '''
        Store useful quanties for generating random faults here
        '''

        super(FaultInjector, self).pre_run(step, level_number)

        if not type(step.levels[level_number].u[0]) == mesh:
            raise NotImplementedError(f'Fault insertion is only implemented for type mesh, not \
{type(step.levels[level_number].u[0])}')

        self.num_nodes = step.levels[0].sweep.params.num_nodes
        self.maxiter = step.params.maxiter
        self.u_shape = step.levels[level_number].u[0].shape
        self.num_levels = len(step.levels)
        self.u_bit_length = 64  # change manually if you ever have something else

        if self.num_levels > 1:
            raise NotImplementedError('I don\'t know how to insert faults in this multi-level madness :(')

        self.timestep_idx = 1
        self.iter_idx = 1

        if self.fault_frequency_time is not np.inf:
            raise NotImplementedError('I can only do faults every so and so many iterations for now')

    def pre_iteration(self, step, level_number):
        '''
        Check we want to flip a bit here
        '''
        super(FaultInjector, self).pre_iteration(step, level_number)

        # check if we want to do a fault now
        if self.timestep_idx % self.fault_frequency_time == 0 or self.iter_idx % self.fault_frequency_iter == 0:
            level, iteration, node, problem_pos, bit = self.generate_random_fault()
            self.inject_fault(step, level_number, level, iteration, node, problem_pos, bit)

        self.iter_idx += 1

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
