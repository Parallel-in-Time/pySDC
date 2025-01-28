import firedrake as fd

from gusto.time_discretisation.time_discretisation import TimeDiscretisation, wrapper_apply
from gusto.core.labels import explicit

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, GenericGustoImex
from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted

import logging
import numpy as np


class LogTime(Hooks):
    """
    Utility hook for knowing how far we got when using adaptive step size selection.
    """

    def post_step(self, step, level_number):
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=-1,
            iter=-1,
            sweep=-1,
            type='_time',
            value=L.time + L.dt,
        )


class pySDC_integrator(TimeDiscretisation):
    """
    This class can be entered into Gusto as a time discretization scheme and will solve steps using pySDC.
    It will construct a pySDC controller which can be used by itself and will be used within the time step when called
    from Gusto. Access the controller via `pySDC_integrator.controller`. This class also has `pySDC_integrator.stats`,
    which gathers all of the pySDC stats recorded in the hooks during every time step when used within Gusto.

    This class supports subcycling with multi-step SDC. You can use pseudo-parallelism by simply giving `n_steps` > 1 or
    do proper parallelism by giving a `controller_communicator` of kind `pySDC.FiredrakeEnsembleCommunicator` with the
    appropriate size. You also have to toggle between pseudo and proper parallelism with `useMPIController`.
    """

    def __init__(
        self,
        description,
        controller_params,
        domain,
        field_name=None,
        solver_parameters=None,
        options=None,
        imex=False,
        useMPIController=False,
        n_steps=1,
        controller_communicator=None,
    ):
        """
        Initialization

        Args:
            description (dict): pySDC description
            controller_params (dict): pySDC controller params
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            imex (bool): Whether to use IMEX splitting
            useMPIController (bool): Whether to use the pseudo-parallel or proper parallel pySDC controller
            n_steps (int): Number of steps done in parallel when using pseudo-parallel pySDC controller
            controller_communicator (pySDC.FiredrakeEnsembleCommunicator, optional): Communicator for the proper parallel controller
        """

        self._residual = None

        super().__init__(
            domain=domain,
            field_name=field_name,
            solver_parameters=solver_parameters,
            options=options,
        )

        self.description = description
        self.controller_params = controller_params
        self.timestepper = None
        self.dt_next = None
        self.imex = imex
        self.useMPIController = useMPIController
        self.controller_communicator = controller_communicator

        if useMPIController:
            assert (
                type(self.controller_communicator).__name__ == 'FiredrakeEnsembleCommunicator'
            ), f'Need to give a FiredrakeEnsembleCommunicator here, not {type(self.controller_communicator)}'
            if n_steps > 1:
                logging.getLogger(type(self).__name__).warning(
                    f'Warning: You selected {n_steps=}, which will be ignored when using the MPI controller!'
                )
            assert (
                controller_communicator is not None
            ), 'You need to supply a communicator when using the MPI controller!'
            self.n_steps = controller_communicator.size
        else:
            self.n_steps = n_steps

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation, apply_bcs, *active_labels)

        # Check if any terms are explicit
        imex = any(t.has_label(explicit) for t in equation.residual) or self.imex
        if imex:
            self.description['problem_class'] = GenericGustoImex
        else:
            self.description['problem_class'] = GenericGusto

        self.description['problem_params'] = {
            'equation': equation,
            'solver_parameters': self.solver_parameters,
            'residual': self._residual,
            **self.description['problem_params'],
        }
        self.description['level_params']['dt'] = float(self.domain.dt) / self.n_steps

        # add utility hook required for step size adaptivity
        hook_class = self.controller_params.get('hook_class', [])
        if not type(hook_class) == list:
            hook_class = [hook_class]
        hook_class.append(LogTime)
        self.controller_params['hook_class'] = hook_class

        # prepare controller and variables
        if self.useMPIController:
            self.controller = controller_MPI(
                comm=self.controller_communicator,
                description=self.description,
                controller_params=self.controller_params,
            )
        else:
            self.controller = controller_nonMPI(
                self.n_steps, description=self.description, controller_params=self.controller_params
            )

        self.prob = self.level.prob
        self.sweeper = self.level.sweep
        self.x0_pySDC = self.prob.dtype_u(self.prob.init)
        self.t = 0
        self.stats = {}

    @property
    def residual(self):
        """Make sure the pySDC problem residual and this residual are the same"""
        if hasattr(self, 'prob'):
            return self.prob.residual
        else:
            return self._residual

    @residual.setter
    def residual(self, value):
        """Make sure the pySDC problem residual and this residual are the same"""
        if hasattr(self, 'prob'):
            if self.useMPIController:
                self.controller.S.levels[0].prob.residual = value
            else:
                for S in self.controller.MS:
                    S.levels[0].prob.residual = value
        else:
            self._residual = value

    @property
    def step(self):
        """Get the first step on the controller"""
        if self.useMPIController:
            return self.controller.S
        else:
            return self.controller.MS[0]

    @property
    def level(self):
        """Get the finest pySDC level"""
        return self.step.levels[0]

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretization to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x0_pySDC.functionspace.assign(x_in)
        assert np.isclose(
            self.level.params.dt * self.n_steps, float(self.dt)
        ), 'Step sizes have diverged between pySDC and Gusto'

        if self.dt_next is not None:
            assert (
                self.timestepper is not None
            ), 'You need to set self.timestepper to the timestepper in order to facilitate adaptive step size selection here!'
            self.timestepper.dt = fd.Constant(self.dt_next * self.n_steps)
            self.t = self.timestepper.t

        uend, _stats = self.controller.run(u0=self.x0_pySDC, t0=float(self.t), Tend=float(self.t + self.dt))

        # update time variables
        if not np.isclose(self.level.params.dt * self.n_steps, float(self.dt)):
            self.dt_next = self.level.params.dt

        self.t = get_sorted(_stats, type='_time', recomputed=False, comm=self.controller_communicator)[-1][1]

        # update time of the Gusto stepper.
        # After this step, the Gusto stepper updates its time again to arrive at the correct time
        if self.timestepper is not None:
            self.timestepper.t = fd.Constant(self.t - self.dt)

        self.dt = fd.Constant(self.level.params.dt * self.n_steps)

        # update stats and output
        self.stats = {**self.stats, **_stats}
        x_out.assign(uend.functionspace)
