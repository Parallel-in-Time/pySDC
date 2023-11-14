import numpy as np

from pySDC.core.Lagrange import LagrangeApproximation
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.core.Collocation import CollBase


class AdaptiveCollocation(ConvergenceController):
    """
    This convergence controller allows to change the underlying quadrature between iterations.

    Supplying multiple quadrature rules will result in a change to a new quadrature whenever the previous one is
    converged until all methods given are converged, at which point the step is ended.
    Whenever the quadrature is changed, the solution is interpolated from the old nodes to the new nodes to ensure
    accelerated convergence compared to starting from the initial conditions.

    Use this convergence controller by supplying parameters that the sweeper accepts as a list to the `params`.
    For instance, supplying
    ```
    params = {
        'num_nodes': [2, 3],
    }
    ```
    will use collocation methods like you passed to the `sweeper_params` in the `description` object, but will change
    the number of nodes to 2 before the first iteration and to 3 as soon as the 2-node collocation problem is converged.
    This will override whatever you set for the number of nodes in the `sweeper_params`, but you still need to set
    something to allow instantiation of the levels before this convergence controller becomes active.
    Make sure all lists you supply here have the same length.

    Feel free to set `logger_level = 15` in the controller parameters to get comprehensive text output on what exactly
    is happening.

    This convergence controller has various applications.
     - You could try to obtain speedup by doing some inexactness. It is currently possible to set various residual
       tolerances, which will be passed to the levels, corresponding to the accuracy with which each collocation problem
       is solved.
     - You can compute multiple solutions to the same initial value problem with different order. This allows, for
       instance, to do adaptive time stepping.

    When trying to obtain speedup with this, be ware that the interpolation is not for free. In particular, it is
    necessary to reevaluate the right hand side at all nodes afterwards.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Record what variables we want to vary.

        Args:
            controller (pySDC.Controller.controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """

        defaults = {
            'control_order': 300,
            'num_colls': 0,
            'sweeper_params': description['sweeper_params'],
            'vary_keys_sweeper': [],
            'vary_keys_level': [],
        }

        # only these keys can be changed by this convergence controller
        self.allowed_sweeper_keys = ['quad_type', 'num_nodes', 'node_type', 'do_coll_update']
        self.allowed_level_keys = ['restol']

        # add the keys to lists so we know what we need to change later
        for key in params.keys():
            if type(params[key]) == list:
                if key in self.allowed_sweeper_keys:
                    defaults['vary_keys_sweeper'] += [key]
                elif key in self.allowed_level_keys:
                    defaults['vary_keys_level'] += [key]
                else:
                    raise NotImplementedError(f'Don\'t know what to do with key {key} here!')

                defaults['num_colls'] = max([defaults['num_colls'], len(params[key])])

        self.comm = description['sweeper_params'].get('comm', None)
        if self.comm:
            from mpi4py import MPI

            self.MPI_SUM = MPI.SUM

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def matmul(self, A, b):
        """
        Matrix vector multiplication, possibly MPI parallel.
        The parallel implementation performs a reduce operation in every row of the matrix. While communicating the
        entire vector once could reduce the number of communications, this way we never need to store the entire vector
        on any specific rank.

        Args:
            A (2d np.ndarray): Matrix
            b (list): Vector

        Returns:
            List: Axb
        """
        if self.comm:
            res = [A[i, 0] * b[0] if b[i] is not None else None for i in range(A.shape[0])]
            buf = b[0] * 0.0
            for i in range(1, A.shape[1]):
                self.comm.Reduce(A[i, self.comm.rank + 1] * b[self.comm.rank + 1], buf, op=self.MPI_SUM, root=i - 1)
                if i == self.comm.rank + 1:
                    res[i] += buf
            return res
        else:
            return A @ b

    def switch_sweeper(self, S):
        """
        Update to the next sweeper in line.

        Args:
            S (pySDC.Step.step): The current step

        Returns:
            None
        """

        # generate dictionaries with the new parameters
        new_params_sweeper = {
            key: self.params.get(key)[self.status.active_coll] for key in self.params.vary_keys_sweeper
        }
        sweeper_params = self.params.sweeper_params.copy()
        update_params_sweeper = {**sweeper_params, **new_params_sweeper}

        new_params_level = {key: self.params.get(key)[self.status.active_coll] for key in self.params.vary_keys_level}

        # update sweeper for all levels
        for L in S.levels:
            P = L.prob

            # store solution of current level which will be interpolated to new level
            u_old = [me.flatten() if me is not None else me for me in L.u]
            nodes_old = L.sweep.coll.nodes.copy()

            # change sweeper
            L.sweep.__init__(update_params_sweeper)
            L.sweep.level = L

            # reset level to tell it the new structure of the solution
            L.params.__dict__.update(new_params_level)
            L.reset_level(reset_status=False)

            # interpolate solution of old collocation problem to new one
            nodes_new = L.sweep.coll.nodes.copy()
            interpolator = LagrangeApproximation(points=np.append(0, nodes_old))

            u_inter = self.matmul(interpolator.getInterpolationMatrix(np.append(0, nodes_new)), u_old)

            # assign the interpolated values to the nodes in the level
            for i in range(0, len(u_inter)):
                if u_inter[i] is not None:
                    me = P.dtype_u(P.init)
                    me[:] = np.reshape(u_inter[i], P.init[0])
                    L.u[i] = me

            # reevaluate rhs
            for i in range(L.sweep.coll.num_nodes + 1):
                if L.u[i] is not None:
                    L.f[i] = L.prob.eval_f(L.u[i], L.time)

        # log the new parameters
        self.log(f'Switching to collocation {self.status.active_coll + 1} of {self.params.num_colls}', S, level=20)
        msg = 'New quadrature:'
        for key in list(sweeper_params.keys()) + list(new_params_level.keys()):
            if key in self.params.vary_keys_sweeper:
                msg += f'\n--> {key}: {update_params_sweeper[key]}'
            elif key in self.params.vary_keys_level:
                msg += f'\n--> {key}: {new_params_level[key]}'
            else:
                msg += f'\n    {key}: {update_params_sweeper[key]}'
        self.log(msg, S)

    def setup_status_variables(self, controller, **kwargs):
        """
        Add an index for which collocation method to use.

        Args:
            controller (pySDC.Controller.controller): The controller

        Returns:
            None
        """
        self.status = Status(['active_coll'])

    def reset_status_variables(self, controller, **kwargs):
        """
        Reset the status variables between time steps.

        Args:
            controller (pySDC.Controller.controller): The controller

        Returns:
            None
        """
        self.status.active_coll = 0

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Switch to the next collocation method if the current one is converged.

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """
        if (self.status.active_coll < self.params.num_colls - 1) and S.status.done:
            self.status.active_coll += 1
            S.status.done = False
            self.switch_sweeper(S)

    def post_spread_processing(self, controller, S, **kwargs):
        """
        Overwrite the sweeper parameters with the first collocation parameters set up here.

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """
        self.switch_sweeper(S)

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check if we allow the scheme to solve the collocation problems to convergence.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if description["level_params"].get("restol", -1.0) <= 1e-16:
            return (
                False,
                "Switching the collocation problems requires solving them to some tolerance that can be reached. Please set attainable `restol` in the level params",
            )

        return True, ""
