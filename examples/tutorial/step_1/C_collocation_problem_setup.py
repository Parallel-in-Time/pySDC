import numpy as np
import scipy.sparse as sp

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh

from pySDC.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def solve_collocation_problem(prob, coll, dt):

    Q = coll.Qmat[1:, 1:]

    M = sp.eye(prob.nvars * coll.num_nodes) - dt * sp.kron(Q, prob.A)

    u0 = prob.u_exact(t=0)
    uend = prob.u_exact(t=dt)
    u0_coll = np.kron(np.ones(coll.num_nodes), u0.values)

    u_coll = sp.linalg.spsolve(M, u0_coll)

    err = np.linalg.norm(u_coll[-prob.nvars:] - uend.values, np.inf)

    return err

if __name__ == "__main__":

    problem_params = {}
    problem_params['nu'] = 0.1
    problem_params['freq'] = 4
    problem_params['nvars'] = 1023

    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    coll = CollGaussRadau_Right(num_nodes=3, tleft=0, tright=1)

    dt = 0.1

    err = solve_collocation_problem(prob=prob, coll=coll, dt=dt)

    print(err, err <= 4E-04)

    assert err <= 4E-04

