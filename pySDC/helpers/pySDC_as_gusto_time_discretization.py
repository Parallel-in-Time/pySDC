import firedrake as fd

from gusto.time_discretisation.time_discretisation import TimeDiscretisation, wrapper_apply
from gusto.core.labels import explicit

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, GenericGustoImex
from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted


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
    """

    def __init__(
        self,
        equation,
        description,
        controller_params,
        domain,
        field_name=None,
        subcycling_options=None,
        solver_parameters=None,
        limiter=None,
        options=None,
        augmentation=None,
        t0=0,
        imex=False,
    ):
        """
        Initialization

        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation.
            description (dict): pySDC description
            controller_params (dict): pySDC controller params
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """

        self._residual = None

        super().__init__(
            domain=domain,
            field_name=field_name,
            subcycling_options=subcycling_options,
            solver_parameters=solver_parameters,
            limiter=limiter,
            options=options,
            augmentation=augmentation,
        )

        self.description = description
        self.controller_params = controller_params
        self.timestepper = None
        self.dt_next = None
        self.imex = imex

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
        }
        self.description['level_params']['dt'] = float(self.domain.dt)

        # add utility hook required for step size adaptivity
        hook_class = self.controller_params.get('hook_class', [])
        if not type(hook_class) == list:
            hook_class = [hook_class]
        hook_class.append(LogTime)
        self.controller_params['hook_class'] = hook_class

        # prepare controller and variables
        self.controller = controller_nonMPI(1, description=self.description, controller_params=self.controller_params)
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
            self.prob.residual = value
        else:
            self._residual = value

    @property
    def level(self):
        """Get the finest pySDC level"""
        return self.controller.MS[0].levels[0]

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretization to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x0_pySDC.functionspace.assign(x_in)
        assert self.level.params.dt == float(self.dt), 'Step sizes have diverged between pySDC and Gusto'

        if self.dt_next is not None:
            assert (
                self.timestepper is not None
            ), 'You need to set self.timestepper to the timestepper in order to facilitate adaptive step size selection here!'
            self.timestepper.dt = fd.Constant(self.dt_next)
            self.t = self.timestepper.t

        uend, _stats = self.controller.run(u0=self.x0_pySDC, t0=float(self.t), Tend=float(self.t + self.dt))

        # update time variables
        if self.level.params.dt != float(self.dt):
            self.dt_next = self.level.params.dt

        self.t = get_sorted(_stats, type='_time', recomputed=False)[-1][1]

        # update time of the Gusto stepper.
        # After this step, the Gusto stepper updates its time again to arrive at the correct time
        if self.timestepper is not None:
            self.timestepper.t = fd.Constant(self.t - self.dt)

        self.dt = self.level.params.dt

        # update stats and output
        self.stats = {**self.stats, **_stats}
        x_out.assign(uend.functionspace)
