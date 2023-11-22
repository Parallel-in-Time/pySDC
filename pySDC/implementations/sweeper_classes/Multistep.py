from pySDC.core.Sweeper import sweeper, _Pars


class Cache(object):
    """
    Class for managing solutions and right hand side evaluations of previous steps for the MultiStep "sweeper".

    Attributes:
        - u (list): Contains solution from previous steps
        - f (list): Contains right hand side evaluations from previous steps
        - t (list): Contains time of previous steps
    """

    def __init__(self, num_steps):
        """
        Initialization routing

        Args:
            num_steps (int): Number of entries for the cache
        """
        self.num_steps = num_steps
        self.u = [None] * num_steps
        self.f = [None] * num_steps
        self.t = [None] * num_steps

    def update(self, t, u, f):
        """
        Add a new value to the cache and remove the oldest one.

        Args:
            t (float): Time of new step
            u (dtype_u): Solution of new step
            f (dtype_f): Right hand side evaluation at new step
        """
        self.u[:-1] = self.u[1:]
        self.f[:-1] = self.f[1:]
        self.t[:-1] = self.t[1:]

        self.u[-1] = u
        self.f[-1] = f
        self.t[-1] = t

    def __str__(self):
        """
        Print the contents of the cache for debugging purposes.
        """
        string = ''
        for i in range(self.num_steps):
            string = f'{string} t={self.t[i]}: u={self.u[i]}, f={self.f[i]}'

        return string


class MultiStep(sweeper):
    alpha = None
    beta = None

    def __init__(self, params):
        """
        Initialization routine for the base sweeper.

        Multistep methods work by constructing Euleresque steps with the right hand side composed of a weighted sum of solutions and right hand side evaluations at previous steps.
        The coefficients in the weighted sum for the solutions are called alpha and the ones for the right hand sides are called beta in this implementation.
        So for an N step method, there need to be N alpha values and N + 1 beta values, where the last one is on the left hand side for implicit methods.
        The first element in the coefficients belongs to the value furthest in the past and vice versa. Values from previous time steps are stored in a `Cache` object.
        Be careful with the sign of the alpha values. You can look at the implementations of the Euler methods for guidance.

        Class attributes:
            alpha (list): Coefficients for the solutions of previous steps
            beta (list): Coefficients for the right hand side evaluations

        Args:
            params (dict): parameter object
        """
        import logging
        from pySDC.core.Collocation import CollBase

        # set up logger
        self.logger = logging.getLogger('sweeper')

        self.params = _Pars(params)

        # check if some parameters are set which only apply to actual sweepers
        for key in ['initial_guess', 'collocation_class', 'num_nodes', 'quad_type']:
            if key in params:
                self.logger.warning(f'"{key}" will be ignored by multistep sweeper')

        # we need a dummy collocation object to instantiate the levels.
        self.coll = CollBase(num_nodes=1, quad_type='RADAU-RIGHT')

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        self.parallelizable = False

        # proprietary variables for the multistep methods
        self.steps = len(self.alpha)
        self.cache = Cache(self.steps)

    def predict(self):
        """
        Add the initial conditions to the cache if needed.

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """
        lvl = self.level

        if all(me is None for me in self.cache.t):
            prob = lvl.prob
            lvl.f[0] = prob.eval_f(lvl.u[0], lvl.time)
            self.cache.update(lvl.time, lvl.u[0], lvl.f[0])

        lvl.status.unlocked = True
        lvl.status.updated = True

    def compute_residual(self, stage=None):
        """
        Do nothing.

        Args:
            stage (str): The current stage of the step the level belongs to
        """
        lvl = self.level
        lvl.status.residual = 0.0
        lvl.status.updated = False

        return None

    def compute_end_point(self):
        """
        The solution is stored in the single node that we have.
        """
        self.level.uend = self.level.u[-1]

    def update_nodes(self):
        """
        Compute the solution to the current step. If the cache is not filled, we compute a provisional solution with a different method.
        """

        lvl = self.level
        prob = lvl.prob
        time = lvl.time + lvl.dt

        if None in self.cache.t:
            self.generate_starting_values()

        else:
            # build the right hand side from the previous solutions
            rhs = prob.dtype_u(prob.init)
            dts = [self.cache.t[i + 1] - self.cache.t[i] for i in range(self.steps - 1)] + [
                time - self.cache.t[-1]
            ]  # TODO: Does this work for adaptive step sizes?
            for i in range(len(self.alpha)):
                rhs -= self.alpha[i] * self.cache.u[i]
                rhs += dts[i] * self.beta[i] * self.cache.f[i]

            # compute the solution to the current step "implicit Euler style"
            lvl.u[1] = prob.solve_system(rhs, lvl.dt * self.beta[-1], self.cache.u[-1], time)

        lvl.f[1] = prob.eval_f(lvl.u[1], time)
        self.cache.update(time, lvl.u[1], lvl.f[1])

    def generate_starting_values(self):
        """
        Compute solutions to the steps when not enough previous values are available for the multistep method.
        The initial conditions are added in `predict` since this is not bespoke behaviour to any method.
        """
        raise NotImplementedError(
            "No implementation for generating solutions when not enough previous values are available!"
        )


class AdamsBashforthExplicit1Step(MultiStep):
    """
    This is just forward Euler.
    """

    alpha = [-1.0]
    beta = [1.0, 0.0]


class BackwardEuler(MultiStep):
    """
    Almost as old, impressive and beloved as Koelner Dom.
    """

    alpha = [-1.0]
    beta = [0.0, 1.0]


class AdamsMoultonImplicit1Step(MultiStep):
    """
    Trapezoidal method dressed up as a multistep method.
    """

    alpha = [-1.0]
    beta = [0.5, 0.5]


class AdamsMoultonImplicit2Step(MultiStep):
    """
    Third order implicit scheme
    """

    alpha = [0.0, -1.0]
    beta = [-1.0 / 12.0, 8.0 / 12.0, 5.0 / 12.0]

    def generate_starting_values(self):
        """
        Generate starting value by trapezoidal rule.
        """
        lvl = self.level
        prob = lvl.prob
        time = lvl.time + lvl.dt

        # do trapezoidal rule
        rhs = lvl.u[0] + lvl.dt / 2 * lvl.f[0]

        lvl.u[1] = prob.solve_system(rhs, lvl.dt / 2.0, lvl.u[0], time)
