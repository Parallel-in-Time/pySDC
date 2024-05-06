#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem class for dedalus
"""
from pySDC.core.Problem import ptype

import numpy as np
from scipy.linalg import blas
from collections import deque

import dedalus.public as d3
from dedalus.core.system import CoeffSystem
from dedalus.tools.array import csr_matvecs


class DedalusProblem(ptype):

    dtype_u = CoeffSystem
    dtype_f = CoeffSystem

    # Dummy class to trick Dedalus
    class DedalusTimeStepper:
        steps = 1
        stages = 1
        def __init__(self, solver):
            self.solver = solver

    def __init__(self, problem:d3.IVP, nNodes, collUpdate=False):
        solver = problem.build_solver(self.DedalusTimeStepper)
        self.solver = solver

        self.M = nNodes

        c = lambda: CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.MX0, self.RHS = c(), c()
        self.LX = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[c() for _ in range(self.M)] for _ in range(2)])

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

    def stateCopy(self):
        return [u.copy() for u in self.solver.state]


    def computeMX0(self, state, MX0):
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


    def updateLHS(self, dt, qI, init=False):
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
        # Update only if different dt
        if self.dt == dt:
            return

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


    def evalLX(self, LX):
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

        self.requireStateCoeffSpace(solver.state)

        # Evaluate matrix vector product and store
        LX.data.fill(0)
        for sp in solver.subproblems:
            spX = sp.gather(solver.state)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))


    def evalF(self, F, time, dt, wall_time):
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

    def solveAndStoreState(self, iNode):
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

        self.presetStateCoeffSpace(solver.state)

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


    def requireStateCoeffSpace(self, state):
        """Transform current state fields in coefficient space.
        If already in coefficient space, doesn't do anything."""
        for field in state:
            field.require_coeff_space()


    def presetStateCoeffSpace(self, state):
        """Allow to write fields in coefficient space into current state
        fields, without transforming current state in coefficient space."""
        for field in state:
            field.preset_layout('c')
