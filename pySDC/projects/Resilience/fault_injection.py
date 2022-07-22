import struct
import numpy as np

from pySDC.core.Hooks import hooks
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.helpers.pysdc_helper import FrozenClass


class Fault(FrozenClass):
    '''
    Class for storing all the data that belongs to a fault, i.e. when and where it happens
    '''
    def __init__(self, params=None):
        '''
        Initialization routine for faults

        Args:
            params (dict): Parameters regarding when the fault will be inserted
        '''

        params = {} if params is None else params

        self.time = None
        self.timestep = None
        self.level_number = None
        self.iteration = None
        self.node = None
        self.problem_pos = None
        self.bit = None
        self.target = 0
        self.when = 'after'  # before or after an iteration?
        self.happened = False

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze

    @classmethod
    def random(cls, args, rnd_params, random_generator=None):
        '''
        Classmethod to initialize a random fault

        Args:
            args (dict): Supply variables that will be exempt from randomization here
            rnd_params (dict): Supply attributes to the randomization such as maximum values here
            random_generator (numpy.random.RandomState): Give a random generator to ensure repeatability
        '''

        if random_generator is None:
            random_generator = np.random.RandomState(2187)

        random = {
            'level_number': random_generator.randint(low=0, high=rnd_params['level_number']),
            'node': random_generator.randint(low=0, high=rnd_params['node'] + 1),
            'iteration': random_generator.randint(low=1, high=rnd_params['iteration'] + 1),
            'problem_pos': [random_generator.randint(low=0, high=i) for i in rnd_params['problem_pos']],
            'bit': random_generator.randint(low=0, high=rnd_params['bit']),
        }

        return cls({**random, **args})


class FaultInjector(hooks):
    '''
    Class to use as base for hooks class instead of abstract hooks class to insert faults using hooks
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(FaultInjector, self).__init__()
        self.fault_frequency_time = np.inf
        self.fault_frequency_iter = np.inf
        self.faults = []
        self.fault_init = []  # add faults to this list when the random parameters have not been set up yet
        self.rnd_params = {}
        self.random_generator = np.random.RandomState(2187)  # number of the cell in which Princess Leia is held

    def add_random_fault(self, args=None, rnd_args=None):
        '''
        Method to generate a random fault and add it to the list of faults to be injected at some point

        Args:
            args (dict): parameters for fault initialization that should not be randomized
            rnd_args (dict): special parameters for randomization other than the default ones

        Returns:
            None
        '''

        # replace args and rnd_args with empty dict if we didn't specify anything
        args = {} if args is None else args
        rnd_args = {} if rnd_args is None else rnd_args

        # check if we can add the fault directly, or if we have to store its parameters and add it in the pre_run hook
        if self.rnd_params == {}:
            self.fault_init += [{'args': args, 'rnd_args': rnd_args}]
        else:
            self.faults += [Fault.random(args=args, rnd_params={**self.rnd_params, **rnd_args},
                            random_generator=self.random_generator)]

        return None

    def inject_fault(self, step, f):
        '''
        Method to inject a fault into a step.

        Args:
            step (pySDC.Step.step): Step to inject the fault into
            f (Fault): fault that should be injected

        Returns:
            None
        '''
        self.logger.info(f'Flipping bit {f.bit} {f.when} iteration {f.iteration} in node {f.node}. Target: {f.target}')
        if f.target == 0:
            L = step.levels[f.level_number]
            L.u[f.node][f.problem_pos] = self.flip_bit(L.u[f.node][f.problem_pos][0], f.bit)
            # we need to evaluate the rhs here, because it will happen only after the fault was fixed otherwise
            L.f[f.node] = L.prob.eval_f(L.u[f.node], L.time + L.dt * L.sweep.coll.nodes[max([0, f.node - 1])])
        else:
            raise NotImplementedError(f'Target {f.target} for faults not impelemted!')

        L = step.levels[f.level_number]
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='bitflip',
                          value=(f.level_number, f.iteration, f.node, f.problem_pos, f.bit, f.target))

        f.happened = True
        self.faults.remove(f)

        return None

    def pre_run(self, step, level_number):
        '''
        Setup random parameters and add the faults that we couldn't before here

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        '''

        super(FaultInjector, self).pre_run(step, level_number)

        if not type(step.levels[level_number].u[0]) == mesh:
            raise NotImplementedError(f'Fault insertion is only implemented for type mesh, not \
{type(step.levels[level_number].u[0])}')

        # define parameters for randomization
        self.rnd_params = {
            'level_number': len(step.levels),
            'node': step.levels[0].sweep.params.num_nodes,
            'iteration': step.params.maxiter,
            'problem_pos': step.levels[level_number].u[0].shape,
            'bit': 64,  # change manually if you ever have something else
        }

        # initialize the faults have been added before we knew the random parameters
        for f in self.fault_init:
            self.add_random_fault(args=f['args'], rnd_args=f['rnd_args'])

        if self.rnd_params['level_number'] > 1:
            raise NotImplementedError('I don\'t know how to insert faults in this multi-level madness :(')

        # initialize parameters for periodic fault injection
        self.timestep_idx = 0
        self.iter_idx = 0

        return None

    def pre_step(self, step, level_number):
        '''
        Deal with periodic fault injection here:
          - Increment the index for counting time steps
          - Add a random fault in this time step if it is time for it based on the frequency

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        '''

        super(FaultInjector, self).pre_step(step, level_number)

        self.timestep_idx += 1

        if self.timestep_idx % self.fault_frequency_time == 0 and not self.timestep_idx == 0:
            self.add_random_fault(args={'timestep': self.timestep_idx})

        return None

    def pre_iteration(self, step, level_number):
        '''
        Check if we have a fault that should be inserted here and deal with periodic injection per iteration count

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        '''

        super(FaultInjector, self).pre_iteration(step, level_number)

        # check if the fault-free iteration count period has elapsed
        if self.iter_idx % self.fault_frequency_iter == 0 and not self.iter_idx == 0:
            self.add_random_fault(args={'timestep': self.timestep_idx, 'iteration': step.status.iter})

        # loop though all unhappened faults and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'before']:
            # based on iteration number
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            # based on time
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration:
                    self.inject_fault(step, f)

        self.iter_idx += 1

        return None

    def post_iteration(self, step, level_number):
        '''
        Check if we have a fault that should be inserted here

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        '''

        super(FaultInjector, self).post_iteration(step, level_number)

        # loop though all unhappened faults and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'after']:
            # based on iteration number
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            # based on time
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration:
                    self.inject_fault(step, f)

        return None

    def to_binary(self, f):
        '''
        Converts a single float in a string containing its binary representation in memory following IEEE754
        The struct.pack function returns the input with the applied conversion code in 8 bit blocks, which are then
        concatenated as a string

        Args:
            f (float, np.float64, np.float32): number to be converted to binary representation

        Returns:
            (str) Binary representation of f following IEEE754 as a string
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

        Args:
            s (str): binary representation of a float number of 32 or 64 bit length following IEEE754

        Returns:
            (float) floating point representation of the binary string
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

        Args:
            target (float, np.float64, np.float32): the floating point number in which you want to flip a bit
            bit (int): the bit which you intend to flip

        Returns:
            (float) The floating point number resulting from flipping the respective bit in target
        '''
        binary = self.to_binary(target)
        return self.to_float(f'{binary[:bit]}{int(binary[bit]) ^ 1}{binary[bit+1:]}')
