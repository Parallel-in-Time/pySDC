import struct
import numpy as np

from pySDC.core.Hooks import hooks
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.helpers.pysdc_helper import FrozenClass


def get_combination_from_index(index, options):
    """
    Transform an index into a set of combinations. This is used when trying all possible combinations for fault insertion.

    For instance if you want to insert a fault in any iteration in any node than you have k options for iterations and M
    options for the node for a total of M * k possibilities. You can then pass an index between 0 and M * k to this
    function together with the options [k, M] which will return a unique value for both k and M from the index. For
    instance, index = 0 will return [0, 0] and index = k * M will return [k-1, M-1].

    Args:
        index (int): Index of the combination
        options (list): The number of options for the each combination.
    """

    if len(options) == 1:
        return [index % options[0]]
    else:
        return [index % options[0]] + get_combination_from_index(index // options[0], options[1:])


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
        self.rank = None
        self.target = 0
        self.when = 'after'  # before or after an iteration?

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

        Returns Fault: Randomly generated fault
        '''

        if random_generator is None:
            random_generator = np.random.RandomState(2187)

        random = {
            'level_number': random_generator.randint(low=0, high=rnd_params['level_number']),
            'node': random_generator.randint(low=rnd_params.get('min_node', 0), high=rnd_params['node'] + 1),
            'iteration': random_generator.randint(low=1, high=rnd_params['iteration'] + 1),
            'problem_pos': [random_generator.randint(low=0, high=i) for i in rnd_params['problem_pos']],
            'bit': random_generator.randint(low=0, high=rnd_params['bit']),
            'rank': random_generator.randint(low=0, high=rnd_params['rank']),
        }
        return cls({**random, **args})

    @classmethod
    def index_to_combination(cls, args, rnd_params, generator=None):
        '''
        Classmethod to initialize a fault based on an index to translate to a combination of fault parameters, in order
        to loop through all combinations. Probably only makes sense for ODEs.

        First, we get the number of possible combinations m, and then get a value for each fault parameter as
        i = m % i_max (plus modifications to make sure we get a sensible value)

        Args:
            args (dict): Supply variables that will be exempt from randomization here.
            rnd_params (dict): Supply attributes to the randomization such as maximum values here
            generator (int): Index for specific combination

        Returns:
            Fault: Generated from a specific combination of parameters
        '''

        ranges = [
            (0, rnd_params['level_number']),
            (rnd_params.get('min_node', 0), rnd_params['node'] + 1),
            (1, rnd_params['iteration'] + 1),
            (0, rnd_params['bit']),
            (0, rnd_params['rank']),
        ]
        ranges += [(0, i) for i in rnd_params['problem_pos']]

        # get values for taking modulo later
        mods = [me[1] - me[0] for me in ranges]

        # get the combinations from the index
        combinations = get_combination_from_index(generator, mods)

        # translate the combinations into a fault that we want to add
        combination = {
            'level_number': range(*ranges[0])[combinations[0]],
            'node': range(*ranges[1])[combinations[1]],
            'iteration': range(*ranges[2])[combinations[2]],
            'bit': range(*ranges[3])[combinations[3]],
            'rank': range(*ranges[4])[combinations[4]],
            'problem_pos': [range(*ranges[5])[combinations[5 + i]] for i in range(len(rnd_params['problem_pos']))],
        }

        return cls({**combination, **args})


class FaultInjector(hooks):
    '''
    Class to use as base for hooks class instead of abstract hooks class to insert faults using hooks
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super().__init__()
        self.fault_frequency_time = np.inf
        self.fault_frequency_iter = np.inf
        self.faults = []
        self.fault_init = []  # add faults to this list when the random parameters have not been set up yet
        self.rnd_params = {}
        self.random_generator = np.random.RandomState(2187)  # number of the cell in which Princess Leia is held

    @classmethod
    def generate_fault_stuff_single_fault(
        cls, bit=0, iteration=1, problem_pos=None, level_number=0, node=1, time=None, rank=0
    ):
        """
        Generate a fault stuff object which will insert a single fault at the supplied parameters. Because there will
        be some parameter set for everything, there is no randomization anymore.

        Args:
            bit (int): Which bit to flip
            iteration (int): After which iteration to flip
            problem_pos: Where in the problem to flip a bit, type depends on the problem
            level_number (int): In which level you want to flip
            node (int): In which node to flip
            time (float): The bitflip will occur in the time step after this time is reached
            rank (int): The rank you want to insert the fault into

        Returns:
            dict: Can be supplied to the run functions in the resilience project to generate the single fault
        """
        assert problem_pos is not None, "Please supply a spatial position for the fault as `problem_pos`!"
        assert time is not None, "Please supply a time for the fault as `time`!"
        fault_stuff = {
            'rng': np.random.RandomState(0),
            'args': {
                'bit': bit,
                'iteration': iteration,
                'level_number': level_number,
                'problem_pos': problem_pos,
                'node': node,
                'time': time,
                'rank': rank,
            },
        }
        fault_stuff['rnd_args'] = fault_stuff['args']
        return fault_stuff

    def add_fault(self, args, rnd_args):
        if type(self.random_generator) == int:
            self.add_fault_from_combination(args, rnd_args)
        elif type(self.random_generator) == np.random.RandomState:
            self.add_random_fault(args, rnd_args)
        else:
            raise NotImplementedError(
                f'Don\'t know how to add fault with generator of type \
{type(self.random_generator)}'
            )

    def add_stored_faults(self):
        '''
        Method to add faults that are recorded for later adding in the pre run hook

        Returns:
            None
        '''
        for f in self.fault_init:
            if f['kind'] == 'random':
                self.add_random_fault(args=f['args'], rnd_args=f['rnd_args'])
            elif f['kind'] == 'combination':
                self.add_fault_from_combination(args=f['args'], rnd_args=f['rnd_args'])
            else:
                raise NotImplementedError(f'I don\'t know how to add stored fault of kind {f["kind"]}')

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
            self.fault_init += [{'args': args, 'rnd_args': rnd_args, 'kind': 'random'}]
        else:
            self.faults += [
                Fault.random(
                    args=args, rnd_params={**self.rnd_params, **rnd_args}, random_generator=self.random_generator
                )
            ]

        return None

    def add_fault_from_combination(self, args=None, rnd_args=None):
        '''
        Method to generate a random fault and add it to the list of faults to be injected at some point

        Args:
            args (dict): parameters for fault initialization that override the combinations
            rnd_args (dict): possible values that the parameters can take

        Returns:
            None
        '''

        # replace args and rnd_args with empty dict if we didn't specify anything
        args = {} if args is None else args
        rnd_args = {} if rnd_args is None else rnd_args

        # check if we can add the fault directly, or if we have to store its parameters and add it in the pre_run hook
        if self.rnd_params == {}:
            self.fault_init += [{'args': args, 'rnd_args': rnd_args, 'kind': 'combination'}]
        else:
            self.faults += [
                Fault.index_to_combination(
                    args=args, rnd_params={**self.rnd_params, **rnd_args}, generator=self.random_generator
                )
            ]

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
        L = step.levels[f.level_number]
        _abs_before = None
        _abs_after = None

        # insert the fault in some target
        if f.target == 0:
            '''
            Target 0 means we flip a bit in the solution.

            To make sure the faults have some impact, we have to reevaluate the right hand side. Otherwise the fault is
            fixed automatically in this implementation, as the right hand side is assembled only from f(t, u) and u is
            tempered with after computing f(t, u).

            To be fair to iteration based resilience strategies, we also reevaluate the residual. Otherwise, when a
            fault happens in the last iteration, it will not show up in the residual and the iteration is wrongly
            stopped.
            '''
            _abs_before = abs(L.u[f.node][tuple(f.problem_pos)])
            L.u[f.node][tuple(f.problem_pos)] = self.flip_bit(L.u[f.node][tuple(f.problem_pos)], f.bit)
            L.f[f.node] = L.prob.eval_f(L.u[f.node], L.time + L.dt * L.sweep.coll.nodes[max([0, f.node - 1])])
            L.sweep.compute_residual()
            _abs_after = abs(L.u[f.node][tuple(f.problem_pos)])
        else:
            raise NotImplementedError(f'Target {f.target} for faults not implemented!')

        # log what happened to stats and screen
        self.logger.info(
            f'Flipping bit {f.bit} {f.when} iteration {f.iteration} in node {f.node} on rank {f.rank}. Target: {f.target}. Abs: {_abs_before:.4e} -> {_abs_after:.4e}'
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='bitflip',
            value=(f.level_number, f.iteration, f.node, f.problem_pos, f.bit, f.target, f.rank),
        )

        # remove the fault from the list to make sure it happens only once
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

        super().pre_run(step, level_number)

        if not type(step.levels[level_number].u[0]) == mesh:
            raise NotImplementedError(
                f'Fault insertion is only implemented for type mesh, not \
{type(step.levels[level_number].u[0])}'
            )

        dtype = step.levels[level_number].prob.u_exact(t=0).dtype
        if dtype in [float, np.float64]:
            bit = 64
        elif dtype in [complex]:
            bit = 128
        else:
            raise NotImplementedError(f'Don\'t know how many bits type {dtype} has')

        # define parameters for randomization
        self.rnd_params = {
            'level_number': len(step.levels),
            'node': step.levels[0].sweep.params.num_nodes,
            'iteration': step.params.maxiter,
            'problem_pos': step.levels[level_number].u[0].shape,
            'bit': bit,  # change manually if you ever have something else
            'rank': 0,
            **self.rnd_params,
        }

        # initialize the faults have been added before we knew the random parameters
        if step.status.first:
            self.add_stored_faults()

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
        super().pre_step(step, level_number)

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

        super().pre_iteration(step, level_number)

        # check if the fault-free iteration count period has elapsed
        if self.iter_idx % self.fault_frequency_iter == 0 and not self.iter_idx == 0:
            self.add_random_fault(args={'timestep': self.timestep_idx, 'iteration': step.status.iter})

        # loop though all faults that have not yet happened and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'before']:
            # based on iteration number
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            # based on time
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration and step.status.slot == f.rank:
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

        super().post_iteration(step, level_number)

        # loop though all unhappened faults and check if they are scheduled now
        for f in [me for me in self.faults if me.when == 'after']:
            # based on iteration number
            if self.timestep_idx == f.timestep and step.status.iter == f.iteration:
                self.inject_fault(step, f)
            # based on time
            elif f.time is not None:
                if step.time > f.time and step.status.iter == f.iteration and step.status.slot == f.rank:
                    self.inject_fault(step, f)

        return None

    @classmethod
    def to_binary(cls, f):
        '''
        Converts a single float in a string containing its binary representation in memory following IEEE754
        The struct.pack function returns the input with the applied conversion code in 8 bit blocks, which are then
        concatenated as a string. Complex numbers will be returned as two consecutive strings.

        Args:
            f (float, np.float64, np.float32): number to be converted to binary representation

        Returns:
            (str) Binary representation of f following IEEE754 as a string
        '''
        if type(f) in [np.float64, float]:
            conversion_code = '>d'  # big endian, double
        elif type(f) in [np.float32]:
            conversion_code = '>f'  # big endian, float
        elif type(f) in [np.complex128]:
            return f'{cls.to_binary(f.real)}{cls.to_binary(f.imag)}'
        else:
            raise NotImplementedError(f'Don\'t know how to convert number of type {type(f)} to binary')

        return ''.join('{:0>8b}'.format(c) for c in struct.pack(conversion_code, f))

    @classmethod
    def to_float(cls, s):
        '''
        Converts a string of a IEEE754 binary representation in a float. The string is converted to integer with base 2
        and converted to bytes, which can be unpacked into a Python float by the struct module.

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
        elif len(s) == 128:  # complex floats
            real = s[0:64]
            imag = s[64:128]
            return cls.to_float(real) + cls.to_float(imag) * 1j

        else:
            raise NotImplementedError(f'Don\'t know how to convert string of length {len(s)} to float')

        return struct.unpack(conversion_code, int(s, 2).to_bytes(byte_count, 'big'))[0]

    @classmethod
    def flip_bit(cls, target, bit):
        '''
        Flips a bit at position bit in a target using the bitwise xor operator

        Args:
            target (float, np.float64, np.float32): the floating point number in which you want to flip a bit
            bit (int): the bit which you intend to flip

        Returns:
            (float) The floating point number resulting from flipping the respective bit in target
        '''
        binary = cls.to_binary(target)
        return cls.to_float(f'{binary[:bit]}{int(binary[bit]) ^ 1}{binary[bit+1:]}')


def prepare_controller_for_faults(controller, fault_stuff, rnd_args=None, args=None):
    """
    Prepare the controller for a run with faults. That means the fault injection hook is added and supplied with the
    relevant parameters.

    Args:
        controller (pySDC.controller): The controller
        fault_stuff (dict): A dictionary with information on how to add faults
        rnd_args (dict): Default arguments for how to add random faults in a specific problem
        args (dict): Default arguments for where to add faults in a specific problem

    Returns:
        None
    """
    args = {} if args is None else args
    rnd_args = {} if rnd_args is None else rnd_args

    faultHook = get_fault_injector_hook(controller)
    faultHook.random_generator = fault_stuff['rng']

    for key in ['fault_frequency_iter']:
        if key in fault_stuff.keys():
            faultHook.__dict__[key] = fault_stuff[key]

    if not len(faultHook.rnd_params.keys()) > 0:
        faultHook.add_fault(
            rnd_args={**rnd_args, **fault_stuff.get('rnd_params', {})},
            args={**args, **fault_stuff.get('args', {})},
        )

    for key, val in fault_stuff.get('rnd_params', {}).items():
        faultHook.rnd_params[key] = val

    faultHook.rnd_params['rank'] = {'rank': len(controller.MS), **rnd_args, **fault_stuff.get('rnd_params', {})}.get(
        'rank', 1
    )


def get_fault_injector_hook(controller):
    """
    Get the fault injector hook from the list of hooks in the controller.
    If there is not one already, it is added here.

    Args:
        controller (pySDC.controller): The controller

    Returns:
        pySDC.hook.FaultInjector: The fault injecting hook
    """
    hook_types = [type(me) for me in controller.hooks]

    if FaultInjector not in hook_types:
        controller.add_hook(FaultInjector)
        return get_fault_injector_hook(controller)
    else:
        hook_idx = [i for i in range(len(hook_types)) if hook_types[i] == FaultInjector]
        assert len(hook_idx) == 1, f'Expected exactly one FaultInjector, got {len(hook_idx)}!'
        return controller.hooks[hook_idx[0]]
