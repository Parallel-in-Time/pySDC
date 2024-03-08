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
                    'iL': 0.5393867577881641,
                    'vC': 0.9999999999913842,
                    't_switches': [0.1823215567921536, 0.18232155679215356, 0.18232155679173784],
                    'dt': 0.09453745651144455,
                    'e_em': 1.7587042933087105e-12,
                    'sum_restarts': 15.0,
                    'sum_niters': 280.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5393867577223005,
                    'vC': 0.9999999999813279,
                    't_switches': [0.18232155676894835, 0.1823215567897308, 0.18232155678877865],
                    'dt': 0.06467602356229402,
                    'e_em': 1.1468603844377867e-13,
                    'sum_restarts': 17.0,
                    'sum_niters': 368.0,
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
                    'iL': 0.5456190026495924,
                    'vC': 0.9991666666670524,
                    't_switches': [0.18232155679395579, 0.18232155679395592, 0.18232155679356965],
                    'sum_restarts': 14.0,
                    'sum_niters': 416.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5403849766797957,
                    'vC': 0.9999166666675774,
                    't_switches': [0.18232155679395004, 0.18232155679303919],
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
                    'iL': 0.5393867577468375,
                    'vC': 0.9999999999980123,
                    't_switches': [0.18232155680038617, 0.1823215568023739],
                    'dt': 0.08896232033732146,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 15.0,
                    'sum_niters': 280.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5393867576415584,
                    'vC': 0.9999999999802239,
                    't_switches': [0.18232155678530526, 0.1823215568066914, 0.1823215568057151],
                    'dt': 0.06333183541149384,
                    'e_em': 2.220446049250313e-16,
                    'sum_restarts': 17.0,
                    'sum_niters': 368.0,
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
                    'iL': 0.5490992952561473,
                    'vC': 0.9999999999982524,
                    't_switches': [0.1823215567992934, 0.18232155680104123],
                    'sum_restarts': 14.0,
                    'sum_niters': 416.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.5407340516779595,
                    'vC': 0.9999999999936255,
                    't_switches': [0.18232155676519302],
                    'sum_restarts': 10.0,
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
                    'iL': 0.6125019898578352,
                    'vC1': 1.0000000000471956,
                    'vC2': 1.0000000000165106,
                    't_switches': [0.18232155678158268, 0.36464311353802376],
                    'dt': 0.0985931246953285,
                    'e_em': 2.295386103412511e-12,
                    'sum_restarts': 24.0,
                    'sum_niters': 552.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.6125019901065321,
                    'vC1': 1.0000000000787372,
                    'vC2': 1.000000000028657,
                    't_switches': [0.1823215567907939, 0.3646431134803315],
                    'dt': 0.07154669986159717,
                    'e_em': 2.3381296898605797e-13,
                    'sum_restarts': 22.0,
                    'sum_niters': 472.0,
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
                    'iL': 0.6313858468030417,
                    'vC1': 1.0000000002414198,
                    'vC2': 1.0000000000095093,
                    't_switches': [0.18232155679395579, 0.3646431133464922],
                    'sum_restarts': 19.0,
                    'sum_niters': 664.0,
                }
            elif dt == 1e-3:
                expected = {
                    'iL': 0.6151254295045797,
                    'vC1': 1.0000000000227713,
                    'vC2': 1.0000000000329365,
                    't_switches': [0.18232155680153855, 0.3646431135651182],
                    'sum_restarts': 16.0,
                    'sum_niters': 4224.0,
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
                    'u': 5.998326995729771,
                    'e_global': 0.0041911003550794135,
                    't_switches': [1.6094379124660123],
                    'e_event': [3.1912028575220575e-11],
                    'sum_restarts': 22.0,
                    'sum_niters': 675.0,
                }
            elif dt == 1e-3:
                expected = {
                    'u': 5.9721869476933005,
                    'e_global': 0.0004191100622819022,
                    't_switches': [1.6094379124262566, 1.6094379124260099],
                    'e_event': [7.843725668976731e-12],
                    'sum_restarts': 20.0,
                    'sum_niters': 2352.0,
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
                    'e_global': 0.009855041056925806,
                    'sum_niters': 527.0,
                }
            elif dt == 1e-3:
                expected = {
                    'u': 5.9737411566014105,
                    'e_global': 0.0005763403865515215,
                    'sum_niters': 2226.0,
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
                    'vC1': 9.781955920747619,
                    'vC2': 6.396971204930281,
                    'iLp': -1.1056614708409171,
                    'sum_niters': 2519.0,
                }
            elif dt == 1e-5:
                expected = {
                    'vC1': 9.782142840662102,
                    'vC2': 6.388775533709242,
                    'iLp': -1.0994027552202539,
                    'sum_niters': 4242.0,
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
            if len(expected[key]) == got[key]:
                assert np.allclose(expected[key], got[key], atol=1e-4) == True, err_msg
            else:
                assert np.isclose(expected[key][-1], got[key][-1], atol=1e-4) == True, err_msg
        else:
            err_msg = f'{msg} Expected {key}={expected[key]:.4e}, got {key}={got[key]:.4e}'
            assert np.isclose(expected[key], got[key], atol=1e-4), err_msg
