import struct
import numpy as np

from pySDC.core.Hooks import hooks


class FaultInjector(hooks):

    def __init__(self, seed=0):
        super(FaultInjector, self).__init__()
        self.seed = seed

    def generate_random_fault(self):
        pass

    def inject_fault(self, hook, pos, iteration):
        pass

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
            raise NotImplementedError(f'Don\'nt know how to convert number of type {type(f)} to binary')

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
            raise NotImplementedError(f'Don\'nt know how to convert string of length {len(s)} to float')

        return struct.unpack(conversion_code, int(s, 2).to_bytes(byte_count, 'big'))[0]

    def flip_bit(self, target, bit):
        '''
        Flips a bit at position bit in a target using the bitwise xor operator
        '''
        binary = self.to_binary(target)
        return self.to_float(f'{binary[:bit]}{int(binary[bit]) ^ 1}{binary[bit+1:]}')
