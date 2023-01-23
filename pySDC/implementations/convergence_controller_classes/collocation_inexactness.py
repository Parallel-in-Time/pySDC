import numpy as np

from pySDC.core.Lagrange import LagrangeApproximation
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.core.Collocation import CollBase


class CollocationInexactness(ConvergenceController):
    def setup(self, controller, params, description):
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
        self.allowed_sweeper_keys = ['quad_type', 'num_nodes', 'node_type']
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

        return {**defaults, **params}

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
            u_old = L.u.copy()
            nodes_old = L.sweep.coll.nodes.copy()

            # change sweeper
            L.sweep.__init__(update_params_sweeper)
            L.sweep.level = L

            # reset level to tell it the new structure of the solution
            L.params.__dict__.update(new_params_level)
            L.reset_level(reset_status=False)

            # interpolate solution of old collocation problem to new one
            nodes_new = L.sweep.coll.nodes.copy()
            print(nodes_old, '->', nodes_new)
            interpolator = LagrangeApproximation(points=np.append(0, nodes_old))
            u_inter = interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ u_old

            # assign the interpolated values to the nodes in the level
            for i in range(0, len(u_inter)):
                me = P.dtype_u(P.init)
                me[:] = u_inter[i]
                L.u[i] = me

            # reevaluate rhs
            for i in range(L.sweep.coll.num_nodes + 1):
                L.f[i] = L.prob.eval_f(L.u[i], L.time)

        # log the new parameters
        msg = f'Switching to collocation {self.status.active_coll + 1} of {self.params.num_colls}:'
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
