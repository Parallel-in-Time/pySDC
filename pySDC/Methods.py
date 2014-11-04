from pySDC import Step as stepclass

def check_convergence(S):

        L = S.levels[0]

        res = L.status.residual

        converged = S.iter >= S.params.maxiter or res <= L.params.restol

        L.stats.iter_stats[S.iter-1].residual = res

        if converged:
            S.stats.niter = S.iter
            L.stats.residual = res
            S.stats.residual = res

        return converged


def sdc_step(S):

    assert isinstance(S,stepclass.step)
    assert len(S.levels) == 1

    L = S.levels[0]

    L.sweep.predict()

    S.iter = 0

    converged = False

    while not converged:

        S.iter += 1

        L.stats.add_iter_stats()
        L.sweep.update_nodes()
        L.sweep.compute_residual()
        L.logger.info('Iteration: %2i -- Residual: %12.8e',S.iter,L.status.residual)

        converged = check_convergence(S)

    L.sweep.compute_end_point()

    return L.uend


def mlsdc_step(S):

    assert isinstance(S,stepclass.step)
    assert len(S.levels) > 1

    L = len(S.levels)

    S.iter = 0

    S.levels[0].sweep.predict()

    S.levels[0].stats.add_iter_stats()
    S.levels[0].sweep.update_nodes()
    S.levels[0].sweep.compute_residual()
    S.levels[0].logger.info('Iteration: %2i -- Residual: %12.8e',S.iter,S.levels[0].status.residual)

    converged = check_convergence(S)

    while not converged:

        S.iter += 1

        for l in range(1,L):
            S.transfer(source=S.levels[l-1],target=S.levels[l])
            S.levels[l].sweep.update_nodes()
            S.levels[l].sweep.compute_residual()
            S.levels[l].logger.info('Iteration: %2i -- Residual: %12.8e',S.iter,S.levels[l].status.residual)

        for l in range(L-1,0,-1):
            S.transfer(source=S.levels[l],target=S.levels[l-1])
            S.levels[l-1].stats.add_iter_stats()
            S.levels[l-1].sweep.update_nodes()
            S.levels[l-1].sweep.compute_residual()
            S.levels[l-1].logger.info('Iteration: %2i -- Residual: %12.8e',S.iter,S.levels[l-1].status.residual)

        converged = check_convergence(S)

    S.levels[0].sweep.compute_end_point()

    return S.levels[0].uend


