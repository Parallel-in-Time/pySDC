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
    u_num_tmp = {unknown: u_num[unknown][-1] for unknown in unknowns}

    got = {}
    got = u_num_tmp

    if prob_cls_name == 'battery':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.5393867578149492,
                    'vC': 0.9999999999980306,
                    't_switches': [0.18232155679314516, 0.18232155674931952],
                    'dt': 0.017755006862796352,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 17.0,
                    'sum_niters': 312.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5393867577908672,
                    'vC': 0.9999999999869332,
                    't_switches': [0.182321556754745],
                    'dt': 0.056647727943027276,
                    'e_em': 4.285460875053104e-14,
                    'sum_restarts': 15.0,
                    'sum_niters': 296.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.5490992955276419,
                    'vC': 0.9999999999816187,
                    't_switches': [0.182321556761159],
                    'sum_restarts': 13.0,
                    'sum_niters': 416.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5403849766797957,
                    'vC': 0.9999166666718436,
                    't_switches': [0.18232155679295683, 0.1823215567608114],
                    'sum_restarts': 11.0,
                    'sum_niters': 2536.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.4433805288639916,
                    'vC': 0.90262388393713,
                    'dt': 0.18137307612335937,
                    'e_em': 2.7177844974524135e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.3994744179584864,
                    'vC': 0.9679037468770668,
                    'dt': 0.1701392217033212,
                    'e_em': 2.0992988458701234e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update(
                {
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'iL': 0.5456625861551172,
                    'vC': 0.9973251377556902,
                    'sum_niters': 240.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.538639340748308,
                    'vC': 0.9994050706403905,
                    'sum_niters': 2400.0,
                }
            got.update(
                {
                    'sum_niters': u_num['sum_niters'],
                }
            )

    elif prob_cls_name == 'battery_implicit':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.5393867577386883,
                    'vC': 1.0000000000016669,
                    't_switches': [0.18232155680505413, 0.1823215568065723],
                    'dt': 0.017701859620598615,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 17.0,
                    'sum_niters': 312.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5393867577174891,
                    'vC': 0.9999999999869332,
                    't_switches': [0.18232155677090756],
                    'dt': 0.05664772792686473,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 15.0,
                    'sum_niters': 296.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.549099295497225,
                    'vC': 0.9999999999816187,
                    't_switches': [0.18232155676455783],
                    'sum_restarts': 13.0,
                    'sum_niters': 416.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5403849766692057,
                    'vC': 0.9999166666718439,
                    't_switches': [0.18232155679484557, 0.1823215567627017],
                    'sum_restarts': 11.0,
                    'sum_niters': 2536.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.4694087102919169,
                    'vC': 0.9026238839418407,
                    'dt': 0.18137307612335937,
                    'e_em': 2.3469713394952407e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.39947441811958956,
                    'vC': 0.9679037468856341,
                    'dt': 0.1701392217033212,
                    'e_em': 1.147640815712947e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update(
                {
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'iL': 0.5456915668459889,
                    'vC': 0.9973251377578705,
                    'sum_niters': 240.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5386399890100035,
                    'vC': 0.999405070641239,
                    'sum_niters': 2400.0,
                }
            got.update(
                {
                    'sum_niters': u_num['sum_niters'],
                }
            )

    elif prob_cls_name == 'battery_n_capacitors':
        if use_adaptivity and use_detection:
            msg = f"Error when using switch estimator and adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.6125019898490534,
                    'vC1': 1.000000000007182,
                    'vC2': 1.0000000000479325,
                    't_switches': [0.18232155680297893, 0.36464311356978835],
                    'dt': 0.02973949262636494,
                    'e_em': 3.3306690738754696e-16,
                    'sum_restarts': 24.0,
                    'sum_niters': 496.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.6125019898915002,
                    'vC1': 1.0000000000057128,
                    'vC2': 1.000000000050465,
                    't_switches': [0.1823215567993942, 0.36464311356557344],
                    'dt': 0.0241594380776875,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 24.0,
                    'sum_niters': 504.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.6313858468881018,
                    'vC1': 1.0000000000155491,
                    'vC2': 1.0000000000316576,
                    't_switches': [0.18232155681737047, 0.36464311357240825],
                    'sum_restarts': 20.0,
                    'sum_niters': 680.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.6151254297607596,
                    'vC1': 1.0000000000846776,
                    'vC2': 1.0000000000993976,
                    't_switches': [0.18232155679295683, 0.3646431135032376],
                    'sum_restarts': 17.0,
                    'sum_niters': 4232.0,
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        if use_adaptivity and not use_detection:
            msg = f"Error when using adaptivity for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'iL': 0.15890544838473294,
                    'vC1': 0.8806086293336285,
                    'vC2': 0.9915019206727803,
                    'dt': 0.38137307612335936,
                    'e_em': 4.145817911194172e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 24.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.15422467570971707,
                    'vC1': 0.8756872272783145,
                    'vC2': 0.9971015415168025,
                    'dt': 0.3701392217033212,
                    'e_em': 3.6970297934146856e-09,
                    'sum_restarts': 0.0,
                    'sum_niters': 32.0,
                }
            got.update(
                {
                    'dt': u_num['dt'][-1, 1],
                    'e_em': u_num['e_em'][-1, 1],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                }
            )

        elif not use_adaptivity and not use_detection:
            msg = f'Error for {prob_cls_name} for dt={dt:.1e}:'
            if dt == 1e-2:
                expected = {
                    'iL': 0.5939796175551723,
                    'vC1': 0.9973251377556902,
                    'vC2': 0.9973251377466236,
                    'sum_niters': 400.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.6107051960885036,
                    'vC1': 0.9994050706403905,
                    'vC2': 0.9997382611893499,
                    'sum_niters': 4000.0,
                }
            got.update(
                {
                    'sum_niters': u_num['sum_niters'],
                }
            )

    elif prob_cls_name == 'DiscontinuousTestODE':
        if not use_adaptivity and use_detection:
            msg = f"Error when using switch estimator for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'u': 5.9941358952954955,
                    't_switches': [1.6094379124671208],
                    'sum_restarts': 25.0,
                    'sum_niters': 710.0,
                    'e_global': 8.195133460731086e-11,
                    'e_event': [3.302047524300633e-11],
                }
            elif dt == 1e-3:
                expected = {
                    'u': 5.971767837651004,
                    't_switches': [1.6094379124247695],
                    'sum_restarts': 23.0,
                    'sum_niters': 2388.0,
                    'e_global': 2.3067769916451653e-11,
                    'e_event': [9.330758388159666e-12],
                }
            got.update(
                {
                    't_switches': u_num['t_switches'],
                    'sum_restarts': u_num['sum_restarts'],
                    'sum_niters': u_num['sum_niters'],
                    'e_global': u_num['e_global'][-1, 1],
                    'e_event': u_num['e_event'],
                }
            )

        elif not use_adaptivity and not use_detection:
            msg = f"Error for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 1e-2:
                expected = {
                    'u': 5.9805345175338225,
                    'sum_niters': 527.0,
                    'e_global': 0.009855041056925806,
                }
            elif dt == 1e-3:
                expected = {
                    'u': 5.9737411566014105,
                    'sum_niters': 2226.0,
                    'e_global': 0.0005763403865515215,
                }
        got.update(
            {
                'sum_niters': u_num['sum_niters'],
                'e_global': u_num['e_global'][-1, 1],
            }
        )

    elif prob_cls_name == 'piline':
        if not use_adaptivity and not use_detection:
            msg = f"Error for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 5e-2:
                expected = {
                    'vC1': 83.97535096068474,
                    'vC2': 80.52142367760014,
                    'iLp': 16.096505806313573,
                    'sum_niters': 2045.0,
                }
            elif dt == 1e-2:
                expected = {
                    'vC1': 83.97462422132232,
                    'vC2': 80.52135774600202,
                    'iLp': 16.09884649820726,
                    'sum_niters': 7206.0,
                }
        got.update(
            {
                'sum_niters': u_num['sum_niters'],
            }
        )

    elif prob_cls_name == 'buck_converter':
        if not use_adaptivity and not use_detection:
            msg = f"Error for {prob_cls_name} for dt={dt:.1e}:"
            if dt == 2e-5:
                expected = {
                    'vC1': 9.890997780767632,
                    'vC2': 4.710415385551326,
                    'iLp': -0.315406990615236,
                    'sum_niters': 5036.0,
                }
            elif dt == 1e-5:
                expected = {
                    'vC1': 9.891508522329485,
                    'vC2': 4.70939963429714,
                    'iLp': -0.32177442457657557,
                    'sum_niters': 8262.0,
                }
        got.update(
            {
                'sum_niters': u_num['sum_niters'],
            }
        )

    else:
        raise ParameterError(f"For {prob_cls_name} there is no test implemented here!")

    for key in expected.keys():
        if key == 't_switches' or key == 'e_event':
            err_msg = f'{msg} Expected {key}={expected[key]}, got {key}={got[key]}'
            assert all(np.isclose(expected[key], got[key], atol=1e-4)) == True, err_msg
        else:
            err_msg = f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'
            assert np.isclose(expected[key], got[key], atol=1e-4), err_msg
