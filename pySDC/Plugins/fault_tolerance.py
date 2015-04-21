



iter = None

step = None

def hard_fault_injection(S):
    global iter, step

    if step == S.status.step and iter == S.status.iter:

        print('things went wrong here: step %i -- iteration %i -- time %e' %(S.status.step,S.status.iter,S.status.time))

        S.reset_step()
        S = hard_fault_correction_spread(S)

    return S


def hard_fault_correction_spread(S):

    for l in range(len(S.levels)):

        if not S.status.first:
            ufirst = S.prev.levels[l].prob.dtype_u(S.prev.levels[l].uend)
        else:
            ufirst = S.levels[l].prob.u_exact(S.status.time)

        S.levels[l].u[0] = ufirst
        S.levels[l].sweep.predict()
        S.levels[l].sweep.compute_end_point()


    S.status.stage = 'IT_FINE_SWEEP'

    return S


def hard_fault_correction_interp_all(S):

    for l in range(len(S.levels)):

        if not S.status.first:
            ufirst = S.prev.levels[l].prob.dtype_u(S.prev.levels[l].uend)
        else:
            ufirst = S.levels[l].prob.u_exact(S.status.time)

        if not S.status.last:
            ulast = S.next.levels[l].prob.dtype_u(S.next.levels[l].u[0])
        else:
            ulast = ufirst

        L = S.levels[l]

        L.u[0] = L.dtype_u(ufirst)
        L.f[0] = L.prob.eval_f(L.u[0],L.time)
        for m in range(1,L.sweep.coll.num_nodes+1):
            L.u[m] = (1-L.sweep.coll.nodes[m-1])*ufirst + L.sweep.coll.nodes[m-1]*ulast
            L.f[m] = L.prob.eval_f(L.u[m],L.time+L.dt*L.sweep.coll.nodes[m-1])

        L.status.unlocked = True



                # # L = S.levels[1]
                # #
                # # L.u[0] = cp.deepcopy(uleft1)
                # # L.f[0] = L.prob.eval_f(L.u[0],L.time)
                # # for m in range(1,L.sweep.coll.num_nodes+1):
                # #     L.u[m] = (1-L.sweep.coll.nodes[m-1])*uleft1 + L.sweep.coll.nodes[m-1]*uright1
                # #     L.f[m] = L.prob.eval_f(L.u[m],L.time+L.dt*L.sweep.coll.nodes[m-1])
                # #
                # # L.status.unlocked = True
                # #
                # # S.levels[1].sweep.compute_end_point()
                # # S.levels[0].sweep.compute_end_point()
                #
                #
    # S.transfer(source=S.levels[0],target=S.levels[1])
    # S.levels[1].sweep.update_nodes()
    # S.levels[1].sweep.compute_end_point()
    # S.transfer(source=S.levels[1],target=S.levels[0])
    # S.levels[0].sweep.compute_end_point()
