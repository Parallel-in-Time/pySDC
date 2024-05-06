#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments with dedalus and pySDC
"""
# Base user imports
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3


coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi))
u = dist.Field(name='u', bases=xbasis)

# Initial solution
x = xbasis.local_grid(dist, scale=1)
listK = [0, 1, 2]
u0 = np.sum([np.cos(k*x) for k in listK], axis=0)
np.copyto(u['g'], u0)

plt.figure('Initial solution')
plt.plot(u['g'], label='Real space')
plt.plot(u['c'], 'o', label='Coefficient space')
plt.legend()
plt.grid()

# Problem
dx = lambda f: d3.Differentiate(f, coords['x'])
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) + dx(u) = 0")


# Tools used for pySDC
from pySDC.core.Problem import ptype

import numpy as np
from scipy.linalg import blas
from collections import deque

from dedalus.core.system import CoeffSystem
from dedalus.tools.array import csr_matvecs


class DedalusProblem(ptype):

    dtype_u = CoeffSystem
    dtype_f = CoeffSystem

    class DedalusTimeStepper:
        steps = 1
        stages = 1
        def __init__(self, solver):
            self.solver = solver

    def __init__(self, problem:d3.IVP, nNodes, collUpdate=False):
        solver = problem.build_solver(self.DedalusTimeStepper)
        self.solver = solver

        self.M = nNodes

        self.MX0, self.RHS = self.c, self.c
        self.LX = deque([[self.c for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[self.c for _ in range(self.M)] for _ in range(2)])

        # Attributes
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.dt = None
        self.firstEval = True
        self.isInit = False

        # Instantiate M solver, needed only for collocation update
        if collUpdate:
            for sp in solver.subproblems:
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                else:
                    sp.LHS = sp.M_min @ sp.pre_right
                sp.M_solver = solver.matsolver(sp.LHS, solver)
        self.collUpdate = collUpdate


    @property
    def c(self):
        return CoeffSystem(self.solver.subproblems, dtype=self.solver.dtype)


    def _computeMX0(self, state, MX0):
        """
        Compute MX0 term used in RHS of both initStep and sweep methods

        Update the MX0 attribute of the timestepper object.
        """
        self._requireStateCoeffSpace(state)

        # Compute and store MX0
        MX0.data.fill(0)
        for sp in self.solver.subproblems:
            spX = sp.gather(state)
            csr_matvecs(sp.M_min, spX, MX0.get_subdata(sp))


    def _updateLHS(self, dt, qI, init=False):
        """Update LHS and LHS solvers for each subproblem

        Parameters
        ----------
        dt : float
            Time-step for the updated LHS.
        qI : 2darray
            QDeltaI coefficients.
        init : bool, optional
            Wether or not initialize the LHS_solvers attribute for each
            subproblem. The default is False.
        """
        # Attribute references
        solver = self.solver

        # Update LHS and LHS solvers for each subproblems
        for sp in solver.subproblems:
            if init:
                # Eventually instanciate list of solver (ony first time step)
                sp.LHS_solvers = [None] * self.M
            for i in range(self.M):
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data,
                              sp.M_exp.data + dt*qI[i, i]*sp.L_exp.data)
                else:
                    sp.LHS = (sp.M_min + dt*qI[i, i]*sp.L_min) @ sp.pre_right
                sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)


    def _evalLX(self, LX):
        """
        Evaluate LX using the current state, and store it

        Parameters
        ----------
        LX : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.

        Returns
        -------
        None.

        """
        # Attribute references
        solver = self.solver

        self._requireStateCoeffSpace(solver.state)

        # Evaluate matrix vector product and store
        LX.data.fill(0)
        for sp in solver.subproblems:
            spX = sp.gather(solver.state)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))


    def _evalF(self, F, time, dt, wall_time):
        """
        Evaluate the F operator from the current solver state

        Note
        ----
        After evaluation, state fields are left in grid space

        Parameters
        ----------
        time : float
            Time of evaluation.
        F : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.
        dt : float
            Current time step.
        wall_time : float
            Current wall time.
        """

        solver = self.solver
        # Evaluate non linear term on current state
        t0 = solver.sim_time
        solver.sim_time = time
        if self.firstEval:
            solver.evaluator.evaluate_scheduled(
                wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
            self.firstEval = False
        else:
            solver.evaluator.evaluate_group(
                'F', wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
        # Initialize F with zero values
        F.data.fill(0)
        # Store F evaluation
        for sp in solver.subproblems:
            spX = sp.gather(solver.F)
            csr_matvecs(sp.pre_left, spX, F.get_subdata(sp))
        # Put back initial solver simulation time
        solver.sim_time = t0

    def _solveAndStoreState(self, iNode):
        """
        Solve LHS * X = RHS using the LHS associated to a given node,
        and store X into the solver state.
        It uses the current RHS attribute of the object.

        Parameters
        ----------
        iNode : int
            Index of the nodes.
        """
        # Attribute references
        solver = self.solver
        RHS = self.RHS

        self._presetStateCoeffSpace(solver.state)

        if self.doResidual:
            self._presetStateCoeffSpace(self.U[iNode])

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            # Solve using LHS of the node
            spX = sp.LHS_solvers[iNode].solve(spRHS)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, solver.state)
            if self.doResidual:
                #print('U was set')
                sp.scatter(spX2, self.U[iNode])


    def _requireStateCoeffSpace(self, state):
        """Transform current state fields in coefficient space.
        If already in coefficient space, doesn't do anything."""
        for field in state:
            field.require_coeff_space()


    def _presetStateCoeffSpace(self, state):
        """Allow to write fields in coefficient space into current state
        fields, without transforming current state in coefficient space."""
        for field in state:
            field.preset_layout('c')


    def _sweep(self):
        """Perform a sweep for the current time-step"""
        # Attribute references
        tau, qI, qE, q = self.nodes, self.QDeltaI, self.QDeltaE, self.Q
        solver = self.solver
        t0, dt, wall_time = solver.sim_time, self.dt, self.wall_time
        RHS, MX0 = self.RHS, self.MX0
        Fk, LXk, Fk1, LXk1 = self.F[0], self.LX[0], self.F[1], self.LX[1]
        axpy = self.axpy

        # Loop on all quadrature nodes
        for i in range(self.M):

            # Initialize with MX0 term
            np.copyto(RHS.data, MX0.data)

            # Add quadrature terms
            for j in range(self.M):
                axpy(a=dt*q[i, j], x=Fk[j].data, y=RHS.data)
                axpy(a=-dt*q[i, j], x=LXk[j].data, y=RHS.data)

            # Add F and LX terms from iteration k+1
            for j in range(i):
                axpy(a=dt*qE[i, j], x=Fk1[j].data, y=RHS.data)
                axpy(a=-dt*qI[i, j], x=LXk1[j].data, y=RHS.data)

            # Add F and LX terms from iteration k
            for j in range(i):
                axpy(a=-dt*qE[i, j], x=Fk[j].data, y=RHS.data)
                axpy(a=dt*qI[i, j], x=LXk[j].data, y=RHS.data)
            axpy(a=dt*qI[i, i], x=LXk[i].data, y=RHS.data)

            # Solve system and store node solution in solver state
            self._solveAndStoreState(i)

            # Evaluate and store LX with current state
            self._evalLX(LXk1[i])
            # Evaluate and store F(X, t) with current state
            self._evalF(Fk1[i], t0+dt*tau[i], dt, wall_time)

        # Inverse position for iterate k and k+1 in storage
        # ie making the new evaluation the old for next iteration
        self.F.rotate()
        self.LX.rotate()


# TODO : implement sweeper

# solver = problem.build_solver(timeStepper)

from pySDC.core.Sweeper import sweeper

class DedalusSweeperIMEX(sweeper):

    def __init__(self, params):
        if 'QI' not in params: params['QI'] = 'IE'
        if 'QE' not in params: params['QE'] = 'EE'
        # call parent's initialization routine
        super().__init__(params)
        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.QE = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.QE)

    def update_nodes(self):
        """
        TODO
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(1, M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
