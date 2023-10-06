import numpy as np

from pySDC.core.Errors import ParameterError


def testSolution(u_num, prob_cls_name, dt, use_adaptivity, use_detection):
    r"""
    Test for numerical solution if values satisfy hardcoded values.

    Note
    ----
    Only the items are tested which does make sense for any problem. For instance, `getDataDict` stores the global error
    after each step **for each problem class**. Since an exact solution is only available for e.g. ``DiscontinuousTestODE``,
    a test is only be done for this problem class.

    Hardcoded solutions are computed for only one collocation node ``M_fix``, which is specified for each problem in the
    related files, see ``pySDC.projects.PinTSimE.battery_model``, ``pySDC.projects.PinTSimE.estimation_check`` and
    ``pySDC.projects.PinTSimE.discontinuous_test_ODE``.

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution together with event time found by event detection, step sizes adjusted
        via adaptivity and/or switch estimation.
    prob_cls_name : str
        Indicates which problem class is tested.
    dt : float
        (Initial) step sizes used for the simulation.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    use_detection : bool
        Indicates whether discontinuity handling is used in the simulation or not.
    """

    unknowns = u_num['unknowns']
    u_num_tmp = {unknown:u_num[unknown][-1] for unknown in unknowns}

    got = {}
    got = u_num_tmp

    if prob_cls_name == 'battery':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559718189012,
                    'v_C': 1.0053361988800296,
                    't_switches': [0.18232155679214296],
                    'dt': 0.11767844320785703,
                    'e_em': 7.811640223565064e-12,
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867578949986,
                    'v_C': 1.0000000000165197,
                    't_switches': [0.18232155677793654],
                    'dt': 0.015641173481932502,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559718189012,
                    'v_C': 1.0053361988800296,
                    't_switches': [0.18232155679214296],
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867578949986,
                    'v_C': 1.0000000000165197,
                    't_switches': [0.18232155677793654],
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4433805288639916,
                    'v_C': 0.90262388393713,
                    'dt': 0.18137307612335937,
                    'e_em': 2.7177844974524135e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.3994744179584864,
                    'v_C': 0.9679037468770668,
                    'dt': 0.1701392217033212,
                    'e_em': 2.0992988458701234e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4433805288639916,
                    'v_C': 0.90262388393713,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.3994744179584864,
                    'v_C': 0.9679037468770668,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })

    elif prob_cls_name == 'battery_implicit':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559717904407,
                    'v_C': 1.0053361988803866,
                    't_switches': [0.18232155679736195],
                    'dt': 0.11767844320263804,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867577837699,
                    'v_C': 1.0000000000250129,
                    't_switches': [0.1823215568036829],
                    'dt': 0.015641237833012522,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.5614559717904407,
                    'v_C': 1.0053361988803866,
                    't_switches': [0.18232155679736195],
                    'sum_restarts': 3.0,
                    'sum_niters': 56.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.5393867577837699,
                    'v_C': 1.0000000000250129,
                    't_switches': [0.1823215568036829],
                    'sum_restarts': 14.0,
                    'sum_niters': 328.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4694087102919169,
                    'v_C': 0.9026238839418407,
                    'dt': 0.18137307612335937,
                    'e_em': 2.3469713394952407e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.39947441811958956,
                    'v_C': 0.9679037468856341,
                    'dt': 0.1701392217033212,
                    'e_em': 1.147640815712947e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.4694087102919169,
                    'v_C': 0.9026238839418407,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.39947441811958956,
                    'v_C': 0.9679037468856341,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })

    elif prob_cls_name == 'battery_n_capacitors':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.6244130166029733,
                    'v_C1': 0.999647921822499,
                    'v_C2': 1.0000000000714673,
                    't_switches': [0.18232155679216916, 0.3649951297770592],
                    'dt': 0.01,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 19.0,
                    'sum_niters': 400.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.6112496171462107,
                    'v_C1': 0.9996894956748836,
                    'v_C2': 1.0,
                    't_switches': [0.1823215567907929, 0.3649535697059346],
                    'dt': 0.07298158272977251,
                    'e_em': 2.703393064962256e-13,
                    'sum_restarts': 11.0,
                    'sum_niters': 216.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.6244130166029733,
                    'v_C1': 0.999647921822499,
                    'v_C2': 1.0000000000714673,
                    't_switches': [0.18232155679216916, 0.3649951297770592],
                    'sum_restarts': 19.0,
                    'sum_niters': 400.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.6112496171462107,
                    'v_C1': 0.9996894956748836,
                    'v_C2': 1.0,
                    't_switches': [0.1823215567907929, 0.3649535697059346],
                    'sum_restarts': 11.0,
                    'sum_niters': 216.0,
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'i_L': 0.15890544838473294,
                    'v_C1': 0.8806086293336285,
                    'v_C2': 0.9915019206727803,
                    'dt': 0.38137307612335936,
                    'e_em': 4.145817911194172e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.15422467570971707,
                    'v_C1': 0.8756872272783145,
                    'v_C2': 0.9971015415168025,
                    'dt': 0.3701392217033212,
                    'e_em': 3.6970297934146856e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update({
                'dt': u_num['dt'][-1, 1],
                'e_em': u_num['e_em'][-1, 1],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'i_L': 0.15890544838473294,
                    'v_C1': 0.8806086293336285,
                    'v_C2': 0.9915019206727803,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'i_L': 0.15422467570971707,
                    'v_C1': 0.8756872272783145,
                    'v_C2': 0.9971015415168025,
                    'sum_niters': 32.0,
                }
            got.update({
                'sum_niters': u_num['sum_niters'],
            })

    elif prob_cls_name == 'DiscontinuousTestODE':
        if not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'u': 5.970674257037518,
                    't_switches': [1.6094400125412065],
                    'sum_restarts': 24.0,
                    'sum_niters': 400.0,
                    'e_global': 5.219439376702439e-06,
                    'e_event': [2.1001071062176635e-06],
                }
            elif dt == 1e-3:
                expected = {
                    'u': 5.970674493833128,
                    't_switches': [1.6094399172800302],
                    'sum_restarts': 23.0,
                    'sum_niters': 400.0,
                    'e_global': 4.982643766915373e-06,
                    'e_event': [2.0048459299371046e-06],
                }
            got.update({
                't_switches': u_num['t_switches'],
                'sum_restarts': u_num['sum_restarts'],
                'sum_niters': u_num['sum_niters'],
                'e_global': u_num['e_global'][-1, 1],
                'e_event': u_num['e_event'],
            })

        elif not use_adaptivity and not use_detection:
            msg = f"Error for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'u': 7.581463909607514,
                    'sum_niters': 24.0,
                    'e_global': 1.6107844331306191,
                }
            elif dt == 1e-3:
                expected = {
                    'u': 6.3988856551968105,
                    'sum_niters': 56.0,
                    'e_global': 0.4282061787199156,
                }
        got.update({
            'sum_niters': u_num['sum_niters'],
            'e_global': u_num['e_global'][-1, 1],
        })


    else:
        raise ParameterError(f"For {prob_cls_name} there is no test implemented here!")
    
    for key in expected.keys():
        if key == 't_switches' or key == 'e_event':
            err_msg = f'{msg} Expected {key}={expected[key]}, got {key}={got[key]}'
            assert all(np.isclose(expected[key], got[key], atol=1e-4)) == True, err_msg
        else:
            err_msg = f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'
            assert np.isclose(expected[key], got[key], atol=1e-4), err_msg