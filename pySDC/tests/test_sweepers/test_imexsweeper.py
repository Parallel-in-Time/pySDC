import unittest
import pytest
import numpy as np

node_types = ['EQUID', 'LEGENDRE']
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']


@pytest.mark.base
class TestImexSweeper(unittest.TestCase):
    #
    # Some auxiliary functions which are not tests themselves
    #
    def setupLevelStepProblem(self):
        from pySDC.core import Step as stepclass

        self.description['sweeper_params'] = self.swparams
        step = stepclass.step(description=self.description)
        level = step.levels[0]
        level.status.time = 0.0
        u0 = step.levels[0].prob.u_exact(step.time)
        step.init_step(u0)
        nnodes = step.levels[0].sweep.coll.num_nodes
        problem = level.prob
        return step, level, problem, nnodes

    #
    # General setUp function used by all tests
    #
    def setUp(self):
        from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as imex

        self.pparams = {}
        self.pparams['lambda_s'] = np.array([-0.1 * 1j], dtype='complex')
        self.pparams['lambda_f'] = np.array([-1.0 * 1j], dtype='complex')
        self.pparams['u0'] = np.random.rand()

        self.swparams = {}
        self.swparams['num_nodes'] = 2 + np.random.randint(5)

        lparams = {}
        lparams['dt'] = 1.0

        self.description = {}
        self.description['problem_class'] = swfw_scalar
        self.description['problem_params'] = self.pparams
        self.description['sweeper_class'] = imex
        self.description['level_params'] = lparams

    # ***************
    # **** TESTS ****
    # ***************

    #
    # Check that a level object can be instantiated
    #
    def test_caninstantiate(self):
        from pySDC.core import Step as stepclass
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as imex

        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            self.description['sweeper_params'] = self.swparams
            S = stepclass.step(description=self.description)
            assert isinstance(S.levels[0].sweep, imex), "sweeper in generated level is not an object of type imex"

    #
    # Check that a level object can be registered in a step object (needed as prerequiste to execute update_nodes
    #
    def test_canregisterlevel(self):
        from pySDC.core import Step as stepclass

        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            self.description['sweeper_params'] = self.swparams
            step = stepclass.step(description=self.description)
            L = step.levels[0]
            with self.assertRaises(Exception):
                L.sweep.predict()
            with self.assertRaises(Exception):
                L.update_nodes()
            with self.assertRaises(Exception):
                L.compute_end_point()

    #
    # Check that the sweeper functions update_nodes and compute_end_point can be executed
    #
    def test_canrunsweep(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            self.description['sweeper_params'] = self.swparams
            # After running setupLevelStepProblem, the functions predict, update_nodes and compute_end_point should run
            step, level, problem, nnodes = self.setupLevelStepProblem()
            assert level.u[0] is not None, "After init_step, level.u[0] should no longer be of type None"
            assert level.u[1] is None, "Before predict, level.u[1] and following should be of type None"
            level.sweep.predict()
            # Should now be able to run update nodes
            level.sweep.update_nodes()
            assert level.uend is None, "uend should be None previous to running compute_end_point"
            level.sweep.compute_end_point()
            assert level.uend is not None, "uend still None after running compute_end_point"

    #
    # Make sure a sweep in matrix form is equal to a sweep in node-to-node form
    #
    def test_sweepequalmatrix(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            step, level, problem, nnodes = self.setupLevelStepProblem()
            step.levels[0].sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Perform node-to-node SDC sweep
            level.sweep.update_nodes()

            lambdas = [problem.lambda_f[0], problem.lambda_s[0]]
            LHS, RHS = level.sweep.get_scalar_problems_sweeper_mats(lambdas=lambdas)

            unew = np.linalg.inv(LHS).dot(u0full + RHS.dot(u0full))
            usweep = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])
            assert (
                np.linalg.norm(unew - usweep, np.infty) < 1e-14
            ), "Single SDC sweeps in matrix and node-to-node formulation yield different results"

    #
    # Make sure the implemented update formula matches the matrix update formula
    #
    def test_updateformula(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            step, level, problem, nnodes = self.setupLevelStepProblem()
            level.sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Perform update step in sweeper
            level.sweep.update_nodes()
            ustages = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])
            # Compute end value through provided function
            level.sweep.compute_end_point()
            uend_sweep = level.uend
            # Compute end value from matrix formulation
            if level.sweep.params.do_coll_update:
                uend_mat = self.pparams['u0'] + step.dt * level.sweep.coll.weights.dot(
                    ustages * (problem.lambda_s[0] + problem.lambda_f[0])
                )
            else:
                uend_mat = ustages[-1]
            assert (
                np.linalg.norm(uend_sweep - uend_mat, np.infty) < 1e-14
            ), "Update formula in sweeper gives different result than matrix update formula"

    #
    # Compute the exact collocation solution by matrix inversion and make sure it is a fixed point
    #
    def test_collocationinvariant(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            step, level, problem, nnodes = self.setupLevelStepProblem()
            level.sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            QE, QI, Q = level.sweep.get_sweeper_mats()

            # Build collocation matrix
            Mcoll = np.eye(nnodes) - step.dt * Q * (problem.lambda_s[0] + problem.lambda_f[0])

            # Solve collocation problem directly
            ucoll = np.linalg.inv(Mcoll).dot(u0full)

            # Put stages of collocation solution into level
            for l in range(0, nnodes):
                level.u[l + 1][:] = ucoll[l]
                level.f[l + 1].impl[:] = problem.lambda_f[0] * ucoll[l]
                level.f[l + 1].expl[:] = problem.lambda_s[0] * ucoll[l]

            # Perform node-to-node SDC sweep
            level.sweep.update_nodes()

            lambdas = [problem.lambda_f[0], problem.lambda_s[0]]
            LHS, RHS = level.sweep.get_scalar_problems_sweeper_mats(lambdas=lambdas)

            # Make sure both matrix and node-to-node sweep leave collocation unaltered
            unew = np.linalg.inv(LHS).dot(u0full + RHS.dot(ucoll))
            assert (
                np.linalg.norm(unew - ucoll, np.infty) < 1e-14
            ), "Collocation solution not invariant under matrix SDC sweep"
            unew_sweep = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])
            print(np.linalg.norm(unew_sweep - ucoll, np.infty))
            assert (
                np.linalg.norm(unew_sweep - ucoll, np.infty) < 1e-14
            ), "Collocation solution not invariant under node-to-node sweep"

    #
    # Make sure that K node-to-node sweeps give the same result as K sweeps in matrix form and the single matrix formulation for K sweeps
    #
    def test_manysweepsequalmatrix(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            step, level, problem, nnodes = self.setupLevelStepProblem()
            step.levels[0].sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Perform K node-to-node SDC sweep
            K = 1 + np.random.randint(6)
            for i in range(0, K):
                level.sweep.update_nodes()
            usweep = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            lambdas = [problem.lambda_f[0], problem.lambda_s[0]]
            LHS, RHS = level.sweep.get_scalar_problems_sweeper_mats(lambdas=lambdas)

            unew = u0full
            for i in range(0, K):
                unew = np.linalg.inv(LHS).dot(u0full + RHS.dot(unew))

            assert (
                np.linalg.norm(unew - usweep, np.infty) < 1e-14
            ), "Doing multiple node-to-node sweeps yields different result than same number of matrix-form sweeps"

            Mat_sweep = level.sweep.get_scalar_problems_manysweep_mat(nsweeps=K, lambdas=lambdas)
            usweep_onematrix = Mat_sweep.dot(u0full)
            assert (
                np.linalg.norm(usweep_onematrix - usweep, np.infty) < 1e-14
            ), "Single-matrix multiple sweep formulation yields different result than multiple sweeps in node-to-node or matrix form form"

    #
    # Make sure that update function for K sweeps computed from K-sweep matrix gives same result as K sweeps in node-to-node form plus compute_end_point
    #
    def test_manysweepupdate(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            step, level, problem, nnodes = self.setupLevelStepProblem()
            step.levels[0].sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Perform K node-to-node SDC sweep
            K = 1 + np.random.randint(6)
            for i in range(0, K):
                level.sweep.update_nodes()
            # Fetch final value
            level.sweep.compute_end_point()
            uend_sweep = level.uend

            lambdas = [problem.lambda_f[0], problem.lambda_s[0]]

            # Build single matrix representing K sweeps
            Mat_sweep = level.sweep.get_scalar_problems_manysweep_mat(nsweeps=K, lambdas=lambdas)
            # Now build update function
            if level.sweep.params.do_coll_update:
                update = 1.0 + (problem.lambda_s[0] + problem.lambda_f[0]) * level.sweep.coll.weights.dot(
                    Mat_sweep.dot(np.ones(nnodes))
                )
                # Multiply u0 by value of update function to get end value directly
                uend_matrix = update * self.pparams['u0']
            else:
                update = Mat_sweep.dot(np.ones(nnodes))
                uend_matrix = (update * self.pparams['u0'])[-1]
            print(abs(uend_matrix - uend_sweep))
            assert (
                abs(uend_matrix - uend_sweep) < 1e-14
            ), "Node-to-node sweep plus update yields different result than update function computed through K-sweep matrix"

    #
    # Make sure the update with do_coll_update=False reproduces last stage
    #
    def test_update_nocollupdate_laststage(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            self.swparams['do_coll_update'] = False
            step, level, problem, nnodes = self.setupLevelStepProblem()
            # if type of nodes does not have right endpoint as quadrature nodes, cannot set do_coll_update to False and perform this test
            if not level.sweep.coll.right_is_node:
                break
            level.sweep.predict()
            ulaststage = np.random.rand()
            level.u[nnodes][:] = ulaststage
            level.sweep.compute_end_point()
            uend = level.uend
            assert (
                abs(uend - ulaststage) < 1e-14
            ), "compute_end_point with do_coll_update=False did not reproduce last stage value"

    #
    # Make sure that update with do_coll_update=False is identical to update formula with q=(0,...,0,1)
    #
    def test_updateformula_no_coll_update(self):
        for node_type, quad_type in zip(node_types, quad_types):
            self.swparams['node_type'] = node_type
            self.swparams['quad_type'] = quad_type
            self.swparams['do_coll_update'] = False
            step, level, problem, nnodes = self.setupLevelStepProblem()
            # if type of nodes does not have right endpoint as quadrature nodes, cannot set do_coll_update to False and perform this test
            if not level.sweep.coll.right_is_node:
                break
            level.sweep.predict()
            u0full = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Perform update step in sweeper
            level.sweep.update_nodes()
            ustages = np.array([level.u[l].flatten() for l in range(1, nnodes + 1)])

            # Compute end value through provided function
            level.sweep.compute_end_point()
            uend_sweep = level.uend
            # Compute end value from matrix formulation
            q = np.zeros(nnodes)
            q[nnodes - 1] = 1.0
            uend_mat = q.dot(ustages)
            assert (
                np.linalg.norm(uend_sweep - uend_mat, np.infty) < 1e-14
            ), "For do_coll_update=False, update formula in sweeper gives different result than matrix update formula with q=(0,..,0,1)"
