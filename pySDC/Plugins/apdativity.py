import numpy as np

def adaptive_predict(Sweeper,Step):
    """
    Predictor to fill values at nodes before first sweep, modified to adapt dt

    Adaptively finds new step sizes by testing a first iteration and checking the residual afterwards. If the
    residual is too high, repeat with dt halved (very basic/heuristic adaptivity).
    """


    # get current level and problem description
    L = Sweeper.level
    P = L.prob

    # evaluate RHS at left point
    L.f[0] = P.eval_f(L.u[0],L.time)

    accepted = False
    while not accepted:

        # copy u[0] to all collocation nodes, evaluate RHS
        for m in range(1,Sweeper.coll.num_nodes+1):
            L.u[m] = P.dtype_u(L.u[0])
            L.f[m] = P.eval_f(L.u[m],L.time+L.dt*Sweeper.coll.nodes[m-1])

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True

        L.sweep.update_nodes()
        L.sweep.compute_residual()

        pred_iter = np.ceil(np.log10(L.status.residual/L.params.restol))
        # print('Predicted niter: ',pred_iter)

        if pred_iter > Step.params.pred_iter_lim:
            Step.dt = Step.dt/2
            print('Setting dt down to ',Step.dt,Step.time)
        elif pred_iter < Step.params.pred_iter_lim:
            Step.dt = 2*Step.dt
            print('Setting dt up to ',Step.dt, Step.time)
        else:
            accepted = True

    return None
