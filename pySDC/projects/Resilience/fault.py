import numpy as np


class Fault(object):
    '''
    Class for storing all the data that belongs to a fault, i.e. when and where it happens
    Target: 0: f, 1: u
    '''
    def __init__(self, time=None, timestep=None, level_number=None, iteration=None, node=None, problem_pos=None,
                 bit=None, target=0):
        self.time = time
        self.timestep = timestep
        self.level_number = level_number
        self.iteration = iteration
        self.node = node
        self.problem_pos = problem_pos
        self.bit = bit
        self.target = target

        self.happened = False

        if self.target == 0:
            self.when = 'before'
        elif self.target == 1:
            self.when = 'after'
        else:
            raise NotImplementedError(f'I don\'t know when to schedule fault targeting {self.target}!')

    @classmethod
    def random(cls, args, rnd_params, random_generator=None):
        if random_generator is None:
            random_generator = np.random.RandomState(2187)
        random = {
            'level_number': random_generator.randint(low=0, high=rnd_params['level_number']),
            'node': random_generator.randint(low=0, high=rnd_params['node'] + 1),
            'iteration': random_generator.randint(low=1, high=rnd_params['iteration'] + 1),
            'problem_pos': [random_generator.randint(low=0, high=i) for i in rnd_params['problem_pos']],
            'bit': random_generator.randint(low=0, high=rnd_params['bit']),
            'target': random_generator.randint(low=0, high=2)
        }

        return cls(time=args.get('time', None),
                   timestep=args.get('timestep', None),
                   level_number=args.get('level_number', random['level_number']),
                   iteration=args.get('iteration', random['iteration']),
                   node=args.get('node', random['node']),
                   problem_pos=args.get('problem_pos', random['problem_pos']),
                   bit=args.get('bit', random['bit']),
                   target=args.get('target', random['target']))
