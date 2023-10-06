import pytest
import warnings
import numpy as np


@pytest.mark.base
def test_pendulum_u_exact_main():
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = pendulum_2d(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    u_test = prob.u_exact(5.0)


@pytest.mark.base
def test_one_transistor_amplifier_u_exact_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = one_transistor_amplifier(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    u_test = prob.u_exact(5.0)


@pytest.mark.base
def test_two_transistor_amplifier_u_exact_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # instantiate problem
    prob = two_transistor_amplifier(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(8))

    u_test = prob.u_exact(5.0)


#
#   Explicit test for the pendulum example
#
@pytest.mark.base
def test_pendulum_main():
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 5e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 200

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = pendulum_2d
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uend_ref = [0.98613917, -0.16592027, 0.29956023, 1.77825875, 4.82500525]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_one_transistor_amplifier_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-4

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = one_transistor_amplifier
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 2e-2

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uend_ref = [-0.02182035, 3.06674603, 2.89634691, 2.45212382, -2.69727238]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_two_transistor_amplifier_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-4

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = two_transistor_amplifier
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 2e-2

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uend_ref = [
        -5.52721527e-03,
        3.00630407e00,
        2.84974338e00,
        4.07588343e00,
        2.12960582e00,
        2.19430889e00,
        5.89240699e00,
        9.99531182e-02,
    ]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_synchgen_infinite_bus_main():
    from pySDC.projects.DAE.problems.synchronous_machine import synchronous_machine_infinite_bus
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 14

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = synchronous_machine_infinite_bus
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uend_ref = [
        8.30823565e-01,
        -4.02584174e-01,
        1.16966755e00,
        9.47592808e-01,
        -3.68076863e-01,
        -3.87492326e-01,
        -7.77837831e-01,
        -1.67347611e-01,
        1.34810867e00,
        5.46223705e-04,
        1.29690691e-02,
        -8.00823474e-02,
        3.10281509e-01,
        9.94039645e-01,
    ]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_DiscontinuousTestDAE_singularity():
    """
    Test if the the event occurs at the correct time and proves if the right-hand side with the correct values.
    """
    import numpy as np
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE

    t_event = np.arccosh(50.0)
    disc_test_DAE = DiscontinuousTestDAE()

    # test for t < t^* by setting t^* = t^* - eps
    eps = 1e-3
    t_before_event = t_event - eps
    u_before_event = disc_test_DAE.u_exact(t_before_event)
    du_before_event = (np.sinh(t_before_event), np.cosh(t_before_event))
    f_before_event = disc_test_DAE.eval_f(u_before_event, du_before_event, t_before_event)

    assert np.isclose(f_before_event[0], 0.0) and np.isclose(f_before_event[1], 0.0), f"ERROR: Right-hand side after event does not match! Expected {(0.0, 0.0)}, got {f_before_event}"

    # test for t <= t^*
    u_event = disc_test_DAE.u_exact(t_event)
    du_event = (np.sinh(t_event), np.cosh(t_event))
    f_event = disc_test_DAE.eval_f(u_event, du_event, t_event)

    assert np.isclose(f_event[0], 7 * np.sqrt(51.0)) and np.isclose(f_event[1], 0.0), f"ERROR: Right-hand side at event does not match! Expected {(7 * np.sqrt(51), 0.0)}, got {f_event}"

    # test for t > t^* by setting t^* = t^* + eps
    t_after_event = t_event + eps
    u_after_event = disc_test_DAE.u_exact(t_after_event)
    du_after_event = (np.sinh(t_event), np.cosh(t_event))
    f_after_event = disc_test_DAE.eval_f(u_after_event, du_after_event, t_after_event)

    assert np.isclose(f_after_event[0], 7 * np.sqrt(51.0)) and np.isclose(f_after_event[1], 0.0), f"ERROR: Right-hand side after event does not match! Expected {(7 * np.sqrt(51), 0.0)}, got {f_after_event}"


@pytest.mark.base
def test_DiscontinuousTestDAE_SDC():
    """
    Simulates one SDC run for different number of coll.nodes and compares if the error satisfies an approppriate value.
    """

    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    # large errors are expected since the simulation domain contains the event
    err_tol = {
        2: 0.2025,
        3: 0.2308,
        4: 0.2407,
        5: 0.245,
    }

    for M in [2, 3, 4, 5]:
        level_params = {
            'restol' : 1e-13,
            'dt': 1e-1,
        }

        problem_params = {
            'newton_tol': 1e-6,
        }

        sweeper_params = {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': M,
            'QI': 'IE',
        }

        step_params = {
            'maxiter': 45,
        }

        controller_params = {
            'logger_level': 30,
        }

        description = {
            'problem_class': DiscontinuousTestDAE,
            'problem_params': problem_params,
            'sweeper_class': fully_implicit_DAE,
            'sweeper_params': sweeper_params,
            'level_params': level_params,
            'step_params': step_params,
        }

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        t0 = 4.6
        Tend = 4.7

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        uex = P.u_exact(Tend)

        uend, _ = controller.run(u0=uinit, t0=t0, Tend=Tend)

        err = abs(uex[0] - uend[0])
        assert err < err_tol[M], f"ERROR: Error is too large! Expected {err_tol[M]}, got {err}"


@pytest.mark.base
def test_DiscontinuousTestDAE_SDC_detection():
    """
    Test for one SDC run with event detection if the found event is close to the exact value and if the global error
    can be reduced.
    """

    from pySDC.projects.PinTSimE.battery_model import get_recomputed
    from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

    err_tol = {
        2: 5.3952e-9,
        3: 2.6741e-9,
        4: 1.9163e-8,
        5: 2.4791e-8,
    }

    event_err_tol = {
        2: 3.6968e-5,
        3: 1.3496e-8,
        4: 7.6006e-8,
        5: 0.0101,
    }

    for M in [2, 3, 4, 5]:
        level_params = {
            'restol' : 1e-13,
            'dt': 1e-2,
        }

        problem_params = {
            'newton_tol': 1e-6,
        }

        sweeper_params = {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': M,
            'QI': 'IE',
        }

        step_params = {
            'maxiter': 45,
        }

        controller_params = {
            'logger_level': 30,
        }

        switch_estimator_params = {
            'tol': 1e-10,
            'alpha': 0.95,
        }

        restarting_params = {
            'max_restarts': 200,
            'crash_after_max_restarts': False,
        }

        convergence_controllers = {
            SwitchEstimator: switch_estimator_params,
            BasicRestartingNonMPI: restarting_params,
        }

        description = {
            'problem_class': DiscontinuousTestDAE,
            'problem_params': problem_params,
            'sweeper_class': fully_implicit_DAE,
            'sweeper_params': sweeper_params,
            'level_params': level_params,
            'step_params': step_params,
            'convergence_controllers': convergence_controllers,
        }

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        t0 = 4.6
        Tend = 4.7

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        uex = P.u_exact(Tend)

        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        err = abs(uex[0] - uend[0])
        assert err < err_tol[M], f"ERROR for M={M}: Error is too large! Expected {err_tol[M]}, got {err}"

        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'ERROR for M={M}: No events found!'
        t_switches = [item[1] for item in switches]
        t_switch = t_switches[-1]

        t_switch_exact = P.t_switch_exact
        event_err = abs(t_switch_exact - t_switch)
        assert event_err < event_err_tol[M], f"ERROR for M={M}: Event error is too large! Expected {event_err_tol[M]}, got {event_err}"



@pytest.mark.base
def test_WSCC9_SDC_detection():
    """
    Test for one SDC run with event detection if the found event is close to the exact value and if the global error
    can be reduced.
    """
  
    from pySDC.projects.PinTSimE.battery_model import get_recomputed
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.DAE.problems.WSCC9BusSystem import WSCC9BusSystem
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
    from pySDC.implementations.hooks.log_solution import LogSolution

    ref = {
        'Eqp': [1.02565963, 0.87077674, 0.75244422],
        'Si1d': [1.08501824, 0.8659459, 0.59335005],
        'Edp': [-9.89321652e-26, 3.55754231e-01, 5.66358724e-01],
        'Si2q': [0.03994074, -0.380399, -0.67227838],
        'Delta': [-2.13428304, 10.32368025, -1.48474241],
        'w': [370.54298062, 398.85092866, 368.59989826],
        'Efd': [1.33144618, 2.11434102, 2.38996818],
        'RF': [0.22357495, 0.35186554, 0.36373663],
        'VR': [1.3316767, 2.48163506, 2.97000777],
        'TM': [0.98658474, 0.63068939, 1.12527586],
        'PSV': [1., 0.52018862, 1.24497292],
        'Id': [1.03392984e+00, -7.55033973e-36, 1.39602103e+00],
        'Iq': [1.80892723e+00, 1.15469164e-30, 6.38447393e-01],
        'V': [0.97014097, 0.94376174, 0.86739643, 0.9361775, 0.88317809, 0.92201319, 0.83761267, 0.85049254, 0.85661891],
        'TH': [-2.30672821, 9.90481234, -2.45484121, -2.42758466, -2.57057159, -2.4746599, -2.67639373, -2.62752952, -2.5584944],
        't_switch': [0.5937503078440701],
    }

    level_params = {
        'restol' : 5e-13,
        'dt': 1 / (2 ** 5),
    }

    problem_params = {
        'newton_tol': 1e-10,
    }

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'LU',
    }

    step_params = {
        'maxiter': 50,
    }

    controller_params = {
        'logger_level': 30,
        'hook_class': LogSolution,
    }

    switch_estimator_params = {
        'tol': 1e-10,
        'alpha': 0.95,
    }

    restarting_params = {
        'max_restarts': 400,
        'crash_after_max_restarts': False,
    }

    convergence_controllers = {
        SwitchEstimator: switch_estimator_params,
        BasicRestartingNonMPI: restarting_params,
    }

    description = {
        'problem_class': WSCC9BusSystem,
        'problem_params': problem_params,
        'sweeper_class': fully_implicit_DAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 0.7

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    m, n = P.m, P.n

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Eqp = np.array([me[1][0:m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Si1d = np.array([me[1][m:2*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Edp = np.array([me[1][2*m:3*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Si2q = np.array([me[1][3*m:4*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Delta = np.array([me[1][4*m:5*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    w = np.array([me[1][5*m:6*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Efd = np.array([me[1][6*m:7*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    RF = np.array([me[1][7*m:8*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    VR = np.array([me[1][8*m:9*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    TM = np.array([me[1][9*m:10*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    PSV = np.array([me[1][10*m:11*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Id = np.array([me[1][11*m:11*m + m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    Iq = np.array([me[1][11*m + m:11*m + 2*m] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    V = np.array([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]
    TH = np.array([me[1][11*m + 2*m + n:11*m + 2*m + 2*n] for me in get_sorted(stats, type='u', sortby='time')])[-1, :]

    switches = get_recomputed(stats, type='switch', sortby='time')
    assert len(switches) >= 1, "ERROR: No events found!"
    t_switch = np.array([item[1] for item in switches])[-1]

    num = {
        'Eqp': Eqp,
        'Si1d': Si1d,
        'Edp': Edp,
        'Si2q': Si2q,
        'Delta': Delta,
        'w': w,
        'Efd': Efd,
        'RF': RF,
        'VR': VR,
        'TM': TM,
        'PSV': PSV,
        'Id': Id,
        'Iq': Iq,
        'V': V,
        'TH': TH,
        't_switch': t_switch,    
    }

    for key in ref.keys():
        assert all(np.isclose(ref[key], num[key], atol=1e-4)) == True, "For {}: Values not equal! Expected {}, got {}".format(key, ref[key], num[key])