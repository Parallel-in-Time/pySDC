import math
from decimal import Decimal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

from pySDC.implementations.problem_classes.Boussinesq_2D_FD_imex import boussinesq_2d_imex
from pySDC.implementations.problem_classes.boussinesq_helpers.helper_classes import logging, Callback


#
# Runge-Kutta IMEX methods of order 1 to 3
#
class rk_imex:
    def __init__(self, problem, order):

        assert order in [1, 2, 3, 4, 5], "Order must be between 1 and 5"
        self.order = order

        if self.order == 1:
            self.A = np.array([[0, 0], [0, 1]])
            self.A_hat = np.array([[0, 0], [1, 0]])
            self.b = np.array([0, 1])
            self.b_hat = np.array([1, 0])
            self.nstages = 2

        elif self.order == 2:
            self.A = np.array([[0, 0], [0, 0.5]])
            self.A_hat = np.array([[0, 0], [0.5, 0]])
            self.b = np.array([0, 1])
            self.b_hat = np.array([0, 1])
            self.nstages = 2

        elif self.order == 3:
            # parameter from Pareschi and Russo, J. Sci. Comp. 2005
            alpha = 0.24169426078821
            beta = 0.06042356519705
            eta = 0.12915286960590
            self.A_hat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1.0, 0, 0], [0, 1.0 / 4.0, 1.0 / 4.0, 0]])
            self.A = np.array([[alpha, 0, 0, 0], [-alpha, alpha, 0, 0], [0, 1.0 - alpha, alpha, 0],
                               [beta, eta, 0.5 - beta - eta - alpha, alpha]])
            self.b_hat = np.array([0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
            self.b = self.b_hat
            self.nstages = 4

        elif self.order == 4:

            self.A_hat = np.array([[0, 0, 0, 0, 0, 0],
                                   [1. / 2, 0, 0, 0, 0, 0],
                                   [13861. / 62500., 6889. / 62500., 0, 0, 0, 0],
                                   [-116923316275. / 2393684061468., -2731218467317. / 15368042101831.,
                                    9408046702089. / 11113171139209., 0, 0, 0],
                                   [-451086348788. / 2902428689909., -2682348792572. / 7519795681897.,
                                    12662868775082. / 11960479115383., 3355817975965. / 11060851509271., 0, 0],
                                   [647845179188. / 3216320057751., 73281519250. / 8382639484533.,
                                    552539513391. / 3454668386233., 3354512671639. / 8306763924573., 4040. / 17871.,
                                    0]])
            self.A = np.array([[0, 0, 0, 0, 0, 0],
                               [1. / 4, 1. / 4, 0, 0, 0, 0],
                               [8611. / 62500., -1743. / 31250., 1. / 4, 0, 0, 0],
                               [5012029. / 34652500., -654441. / 2922500., 174375. / 388108., 1. / 4, 0, 0],
                               [15267082809. / 155376265600., -71443401. / 120774400., 730878875. / 902184768.,
                                2285395. / 8070912., 1. / 4, 0],
                               [82889. / 524892., 0, 15625. / 83664., 69875. / 102672., -2260. / 8211, 1. / 4]])
            self.b = np.array([82889. / 524892., 0, 15625. / 83664., 69875. / 102672., -2260. / 8211, 1. / 4])
            self.b_hat = np.array([4586570599. / 29645900160., 0, 178811875. / 945068544., 814220225. / 1159782912.,
                                   -3700637. / 11593932., 61727. / 225920.])
            self.nstages = 6

        elif self.order == 5:

            # from Kennedy and Carpenter
            # copied from http://www.mcs.anl.gov/petsc/petsc-3.2/src/ts/impls/arkimex/arkimex.c
            self.A_hat = np.zeros((8, 8))
            getcontext().prec = 56
            self.A_hat[1, 0] = Decimal(41.0) / Decimal(100.0)
            self.A_hat[2, 0] = Decimal(367902744464.) / Decimal(2072280473677.)
            self.A_hat[2, 1] = Decimal(677623207551.) / Decimal(8224143866563.)
            self.A_hat[3, 0] = Decimal(1268023523408.) / Decimal(10340822734521.)
            self.A_hat[3, 1] = 0.0
            self.A_hat[3, 2] = Decimal(1029933939417.) / Decimal(13636558850479.)
            self.A_hat[4, 0] = Decimal(14463281900351.) / Decimal(6315353703477.)
            self.A_hat[4, 1] = 0.0
            self.A_hat[4, 2] = Decimal(66114435211212.) / Decimal(5879490589093.)
            self.A_hat[4, 3] = Decimal(-54053170152839.) / Decimal(4284798021562.)
            self.A_hat[5, 0] = Decimal(14090043504691.) / Decimal(34967701212078.)
            self.A_hat[5, 1] = 0.0
            self.A_hat[5, 2] = Decimal(15191511035443.) / Decimal(11219624916014.)
            self.A_hat[5, 3] = Decimal(-18461159152457.) / Decimal(12425892160975.)
            self.A_hat[5, 4] = Decimal(-281667163811.) / Decimal(9011619295870.)
            self.A_hat[6, 0] = Decimal(19230459214898.) / Decimal(13134317526959.)
            self.A_hat[6, 1] = 0.0
            self.A_hat[6, 2] = Decimal(21275331358303.) / Decimal(2942455364971.)
            self.A_hat[6, 3] = Decimal(-38145345988419.) / Decimal(4862620318723.)
            self.A_hat[6, 4] = Decimal(-1.0) / Decimal(8.0)
            self.A_hat[6, 5] = Decimal(-1.0) / Decimal(8.0)
            self.A_hat[7, 0] = Decimal(-19977161125411.) / Decimal(11928030595625.)
            self.A_hat[7, 1] = 0.0
            self.A_hat[7, 2] = Decimal(-40795976796054.) / Decimal(6384907823539.)
            self.A_hat[7, 3] = Decimal(177454434618887.) / Decimal(12078138498510.)
            self.A_hat[7, 4] = Decimal(782672205425.) / Decimal(8267701900261.)
            self.A_hat[7, 5] = Decimal(-69563011059811.) / Decimal(9646580694205.)
            self.A_hat[7, 6] = Decimal(7356628210526.) / Decimal(4942186776405.)

            self.b_hat = np.zeros(8)
            self.b_hat[0] = Decimal(-872700587467.) / Decimal(9133579230613.)
            self.b_hat[1] = 0.0
            self.b_hat[2] = 0.0
            self.b_hat[3] = Decimal(22348218063261.) / Decimal(9555858737531.)
            self.b_hat[4] = Decimal(-1143369518992.) / Decimal(8141816002931.)
            self.b_hat[5] = Decimal(-39379526789629.) / Decimal(19018526304540.)
            self.b_hat[6] = Decimal(32727382324388.) / Decimal(42900044865799.)
            self.b_hat[7] = Decimal(41.0) / Decimal(200.0)

            self.A = np.zeros((8, 8))
            self.A[1, 0] = Decimal(41.) / Decimal(200.)
            self.A[1, 1] = Decimal(41.) / Decimal(200.)
            self.A[2, 0] = Decimal(41.) / Decimal(400.)
            self.A[2, 1] = Decimal(-567603406766.) / Decimal(11931857230679.)
            self.A[2, 2] = Decimal(41.) / Decimal(200.)
            self.A[3, 0] = Decimal(683785636431.) / Decimal(9252920307686.)
            self.A[3, 1] = 0.0
            self.A[3, 2] = Decimal(-110385047103.) / Decimal(1367015193373.)
            self.A[3, 3] = Decimal(41.) / Decimal(200.)
            self.A[4, 0] = Decimal(3016520224154.) / Decimal(10081342136671.)
            self.A[4, 1] = 0.0
            self.A[4, 2] = Decimal(30586259806659.) / Decimal(12414158314087.)
            self.A[4, 3] = Decimal(-22760509404356.) / Decimal(11113319521817.)
            self.A[4, 4] = Decimal(41.) / Decimal(200.)
            self.A[5, 0] = Decimal(218866479029.) / Decimal(1489978393911.)
            self.A[5, 1] = 0.0
            self.A[5, 2] = Decimal(638256894668.) / Decimal(5436446318841.)
            self.A[5, 3] = Decimal(-1179710474555.) / Decimal(5321154724896.)
            self.A[5, 4] = Decimal(-60928119172.) / Decimal(8023461067671.)
            self.A[5, 5] = Decimal(41.) / Decimal(200.)
            self.A[6, 0] = Decimal(1020004230633.) / Decimal(5715676835656.)
            self.A[6, 1] = 0.0
            self.A[6, 2] = Decimal(25762820946817.) / Decimal(25263940353407.)
            self.A[6, 3] = Decimal(-2161375909145.) / Decimal(9755907335909.)
            self.A[6, 4] = Decimal(-211217309593.) / Decimal(5846859502534.)
            self.A[6, 5] = Decimal(-4269925059573.) / Decimal(7827059040749.)
            self.A[6, 6] = Decimal(41.) / Decimal(200.)
            self.A[7, 0] = Decimal(-872700587467.) / Decimal(9133579230613.)
            self.A[7, 1] = 0.0
            self.A[7, 2] = 0.0
            self.A[7, 3] = Decimal(22348218063261.) / Decimal(9555858737531.)
            self.A[7, 4] = Decimal(-1143369518992.) / Decimal(8141816002931.)
            self.A[7, 5] = Decimal(-39379526789629.) / Decimal(19018526304540.)
            self.A[7, 6] = Decimal(32727382324388.) / Decimal(42900044865799.)
            self.A[7, 7] = Decimal(41.) / Decimal(200.)

            self.b = np.zeros(8)

            self.b[0] = Decimal(-975461918565.) / Decimal(9796059967033.)
            self.b[1] = 0.0
            self.b[2] = 0.0
            self.b[3] = Decimal(78070527104295.) / Decimal(32432590147079.)
            self.b[4] = Decimal(-548382580838.) / Decimal(3424219808633.)
            self.b[5] = Decimal(-33438840321285.) / Decimal(15594753105479.)
            self.b[6] = Decimal(3629800801594.) / Decimal(4656183773603.)
            self.b[7] = Decimal(4035322873751.) / Decimal(18575991585200.)

            self.nstages = 8

        self.problem = problem
        self.ndof = np.shape(problem.M)[0]
        self.logger = logging()
        self.stages = np.zeros((self.nstages, self.ndof))

    def timestep(self, u0, dt):

        # Solve for stages
        for i in range(0, self.nstages):

            # Construct RHS
            rhs = np.copy(u0)
            for j in range(0, i):
                rhs += dt * self.A_hat[i, j] * (self.f_slow(self.stages[j, :])) + \
                    dt * self.A[i, j] * (self.f_fast(self.stages[j, :]))

            # Solve for stage i
            if self.A[i, i] == 0:
                # Avoid call to spsolve with identity matrix
                self.stages[i, :] = np.copy(rhs)
            else:
                self.stages[i, :] = self.f_fast_solve(rhs, dt * self.A[i, i], u0)

        # Update
        for i in range(0, self.nstages):
            u0 += dt * self.b_hat[i] * (self.f_slow(self.stages[i, :])) + \
                dt * self.b[i] * (self.f_fast(self.stages[i, :]))

        return u0

    def f_slow(self, u):
        return self.problem.D_upwind.dot(u)

    def f_fast(self, u):
        return self.problem.M.dot(u)

    def f_fast_solve(self, rhs, alpha, u0):
        cb = Callback()
        sol, info = gmres(self.problem.Id - alpha * self.problem.M, rhs, x0=u0,
                          tol=self.problem.params.gmres_tol_limit, restart=self.problem.params.gmres_restart,
                          maxiter=self.problem.params.gmres_maxiter, callback=cb)
        if alpha != 0.0:
            self.logger.add(cb.getcounter())
        return sol


#
# Trapezoidal rule
#
class trapezoidal:
    def __init__(self, problem, alpha=0.5):
        assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
        self.Ndof = np.shape(problem.M)[0]
        self.order = 2
        self.logger = logging()
        self.problem = problem
        self.alpha = alpha

    def timestep(self, u0, dt):
        B_trap = sp.eye(self.Ndof) + self.alpha * dt * (self.problem.D_upwind + self.problem.M)
        b = B_trap.dot(u0)
        return self.f_solve(b, alpha=(1.0 - self.alpha) * dt, u0=u0)

    #
    # Returns f(u) = c*u
    #
    def f(self, u):
        return self.problem.D_upwind.dot(u) + self.problem.M.dot(u)

    #
    # Solves (Id - alpha*c)*u = b for u
    #
    def f_solve(self, b, alpha, u0):
        cb = Callback()
        sol, info = gmres(self.problem.Id - alpha * (self.problem.D_upwind + self.problem.M), b, x0=u0,
                          tol=self.problem.params.gmres_tol_limit, restart=self.problem.params.gmres_restart,
                          maxiter=self.problem.params.gmres_maxiter, callback=cb)
        if alpha != 0.0:
            self.logger.add(cb.getcounter())
        return sol


#
# A BDF-2 implicit two-step method
#
class bdf2:
    def __init__(self, problem):
        assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
        self.Ndof = np.shape(problem.M)[0]
        self.order = 2
        self.logger = logging()
        self.problem = problem

    def firsttimestep(self, u0, dt):
        return self.f_solve(b=u0, alpha=dt, u0=u0)

    def timestep(self, u0, um1, dt):
        b = (4.0 / 3.0) * u0 - (1.0 / 3.0) * um1
        return self.f_solve(b=b, alpha=(2.0 / 3.0) * dt, u0=u0)

    #
    # Returns f(u) = c*u
    #
    def f(self, u):
        return self.problem.D_upwind.dot(u) + self.problem.M.dot(u)

    #
    # Solves (Id - alpha*c)*u = b for u
    #
    def f_solve(self, b, alpha, u0):
        cb = Callback()
        sol, info = gmres(self.problem.Id - alpha * (self.problem.D_upwind + self.problem.M), b, x0=u0,
                          tol=self.problem.params.gmres_tol_limit, restart=self.problem.params.gmres_restart,
                          maxiter=self.problem.params.gmres_maxiter, callback=cb)
        if alpha != 0.0:
            self.logger.add(cb.getcounter())
        return sol


#
#  Split-Explicit method
#

class SplitExplicit:
    def __init__(self, problem, method, pparams):

        assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
        self.Ndof = np.shape(problem.M)[0]
        self.method = method
        self.logger = logging()
        self.problem = problem
        self.pparams = pparams
        self.NdofTher = 2 * problem.N[0] * problem.N[1]
        self.NdofMom = 2 * problem.N[0] * problem.N[1]

        self.ns = None

        # print("dx  ",problem.h[0])
        # print("dz  ",problem.h[1])

        assert self.method in ["MIS4_4", "RK3"], 'Method must be MIS4_4'

        if self.method == 'RK3':
            self.nstages = 3
            self.aRunge = np.zeros((4, 4))
            self.aRunge[0, 0] = 1. / 3.
            self.aRunge[1, 1] = 1. / 2.
            self.aRunge[2, 2] = 1.
            self.dRunge = np.zeros((4, 4))
            self.gRunge = np.zeros((4, 4))
        if self.method == 'MIS4_4':
            self.nstages = 4
            self.aRunge = np.zeros((4, 4))
            self.aRunge[0, 0] = 0.38758444641450318
            self.aRunge[1, 0] = -2.5318448354142823e-002
            self.aRunge[1, 1] = 0.38668943087310403
            self.aRunge[2, 0] = 0.20899983523553325
            self.aRunge[2, 1] = -0.45856648476371231
            self.aRunge[2, 2] = 0.43423187573425748
            self.aRunge[3, 0] = -0.10048822195663100
            self.aRunge[3, 1] = -0.46186171956333327
            self.aRunge[3, 2] = 0.83045062122462809
            self.aRunge[3, 3] = 0.27014914900250392
            self.dRunge = np.zeros((4, 4))
            self.dRunge[1, 1] = 0.52349249922385610
            self.dRunge[2, 1] = 1.1683374366893629
            self.dRunge[2, 2] = -0.75762080241712637
            self.dRunge[3, 1] = -3.6477233846797109e-002
            self.dRunge[3, 2] = 0.56936148730740477
            self.dRunge[3, 3] = 0.47746263002599681
            self.gRunge = np.zeros((4, 4))
            self.gRunge[1, 1] = 0.13145089796226542
            self.gRunge[2, 1] = -0.36855857648747881
            self.gRunge[2, 2] = 0.33159232636600550
            self.gRunge[3, 1] = -6.5767130537473045E-002
            self.gRunge[3, 2] = 4.0591093109036858E-002
            self.gRunge[3, 3] = 6.4902111640806712E-002
        self.dtRunge = np.zeros(self.nstages)
        for i in range(0, self.nstages):
            self.dtRunge[i] = 0
            temp = 1.
            for j in range(0, i + 1):
                self.dtRunge[i] = self.dtRunge[i] + self.aRunge[i, j]
                temp = temp - self.dRunge[i, j]
            self.dRunge[i, 0] = temp
            for j in range(0, i + 1):
                self.aRunge[i, j] = self.aRunge[i, j] / self.dtRunge[i]
                self.gRunge[i, j] = self.gRunge[i, j] / self.dtRunge[i]

        self.U = np.zeros((self.Ndof, self.nstages + 1))
        self.F = np.zeros((self.Ndof, self.nstages))
        self.FSlow = np.zeros(self.Ndof)
        self.nsMin = 8
        self.logger.nsmall = 0

    def NumSmallTimeSteps(self, dx, dz, dt):

        cs = self.pparams['c_s']
        ns = dt / (.9 / np.sqrt(1 / (dx * dx) + 1 / (dz * dz)) / cs)
        ns = max(np.int(np.ceil(ns)), self.nsMin)
        return ns

    def timestep(self, u0, dt):

        self.U[:, 0] = u0

        self.ns = self.NumSmallTimeSteps(self.problem.h[0], self.problem.h[1], dt)

        for i in range(0, self.nstages):
            self.F[:, i] = self.f_slow(self.U[:, i])
            self.FSlow[:] = 0.
            for j in range(0, i + 1):
                self.FSlow += (self.aRunge[i, j] * self.F[:, j] + self.gRunge[i, j] / dt * (self.U[:, j] - u0))
            self.U[:, i + 1] = 0
            for j in range(0, i + 1):
                self.U[:, i + 1] += self.dRunge[i, j] * self.U[:, j]
            nsLoc = np.int(np.ceil(self.ns * self.dtRunge[i]))
            self.logger.nsmall += nsLoc
            dtLoc = dt * self.dtRunge[i]
            dTau = dtLoc / nsLoc
            self.U[:, i + 1] = self.VerletLin(self.U[:, i + 1], self.FSlow, nsLoc, dTau)
        u0 = self.U[:, self.nstages]
        return u0

    def VerletLin(self, u0, FSlow, ns, dTau):
        for i in range(0, ns):
            u0[0:self.NdofMom] += dTau * (self.f_fastMom(u0) + FSlow[0:self.NdofMom])
            u0[self.NdofMom:self.Ndof] += dTau * (self.f_fastTher(u0) + FSlow[self.NdofMom:self.Ndof])

        return u0

    def RK3Lin(self, u0, FSlow, ns, dTau):

        u = u0
        for i in range(0, ns):
            u = u0 + dTau / 3. * (self.f_fast(u) + FSlow)
            u = u0 + dTau / 2. * (self.f_fast(u) + FSlow)
            u = u0 + dTau * (self.f_fast(u) + FSlow)
            u0 = u

        return u0

    def f_slow(self, u):
        return self.problem.D_upwind.dot(u)

    def f_fast(self, u):
        return self.problem.M.dot(u)

    def f_fastMom(self, u):
        return self.problem.M[0:self.NdofMom, self.NdofMom:self.Ndof].dot(u[self.NdofMom:self.Ndof])

    def f_fastTher(self, u):
        return self.problem.M[self.NdofMom:self.Ndof, 0:self.NdofMom].dot(u[0:self.NdofMom])


class dirk:
    def __init__(self, problem, order):

        assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
        self.Ndof = np.shape(problem.M)[0]
        self.order = order
        self.logger = logging()
        self.problem = problem

        assert self.order in [2, 22, 3, 4, 5], 'Order must be 2,22,3,4'

        if self.order == 2:
            self.nstages = 1
            self.A = np.zeros((1, 1))
            self.A[0, 0] = 0.5
            self.tau = [0.5]
            self.b = [1.0]

        if self.order == 22:
            self.nstages = 2
            self.A = np.zeros((2, 2))
            self.A[0, 0] = 1.0 / 3.0
            self.A[1, 0] = 1.0 / 2.0
            self.A[1, 1] = 1.0 / 2.0

            self.tau = np.zeros(2)
            self.tau[0] = 1.0 / 3.0
            self.tau[1] = 1.0

            self.b = np.zeros(2)
            self.b[0] = 3.0 / 4.0
            self.b[1] = 1.0 / 4.0

        if self.order == 3:
            self.nstages = 2
            self.A = np.zeros((2, 2))
            self.A[0, 0] = 0.5 + 1.0 / (2.0 * math.sqrt(3.0))
            self.A[1, 0] = -1.0 / math.sqrt(3.0)
            self.A[1, 1] = self.A[0, 0]

            self.tau = np.zeros(2)
            self.tau[0] = 0.5 + 1.0 / (2.0 * math.sqrt(3.0))
            self.tau[1] = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))

            self.b = np.zeros(2)
            self.b[0] = 0.5
            self.b[1] = 0.5

        if self.order == 4:
            self.nstages = 3
            alpha = 2.0 * math.cos(math.pi / 18.0) / math.sqrt(3.0)

            self.A = np.zeros((3, 3))
            self.A[0, 0] = (1.0 + alpha) / 2.0
            self.A[1, 0] = -alpha / 2.0
            self.A[1, 1] = self.A[0, 0]
            self.A[2, 0] = (1.0 + alpha)
            self.A[2, 1] = -(1.0 + 2.0 * alpha)
            self.A[2, 2] = self.A[0, 0]

            self.tau = np.zeros(3)
            self.tau[0] = (1.0 + alpha) / 2.0
            self.tau[1] = 1.0 / 2.0
            self.tau[2] = (1.0 - alpha) / 2.0

            self.b = np.zeros(3)
            self.b[0] = 1.0 / (6.0 * alpha * alpha)
            self.b[1] = 1.0 - 1.0 / (3.0 * alpha * alpha)
            self.b[2] = 1.0 / (6.0 * alpha * alpha)

        if self.order == 5:
            self.nstages = 5
            # From Kennedy, Carpenter "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations.
            # A Review"
            self.A = np.zeros((5, 5))
            self.A[0, 0] = 4024571134387. / 14474071345096.

            self.A[1, 0] = 9365021263232. / 12572342979331.
            self.A[1, 1] = self.A[0, 0]

            self.A[2, 0] = 2144716224527. / 9320917548702.
            self.A[2, 1] = -397905335951. / 4008788611757.
            self.A[2, 2] = self.A[0, 0]

            self.A[3, 0] = -291541413000. / 6267936762551.
            self.A[3, 1] = 226761949132. / 4473940808273.
            self.A[3, 2] = -1282248297070. / 9697416712681.
            self.A[3, 3] = self.A[0, 0]

            self.A[4, 0] = -2481679516057. / 4626464057815.
            self.A[4, 1] = -197112422687. / 6604378783090.
            self.A[4, 2] = 3952887910906. / 9713059315593.
            self.A[4, 3] = 4906835613583. / 8134926921134.
            self.A[4, 4] = self.A[0, 0]

            self.b = np.zeros(5)
            self.b[0] = -2522702558582. / 12162329469185.
            self.b[1] = 1018267903655. / 12907234417901.
            self.b[2] = 4542392826351. / 13702606430957.
            self.b[3] = 5001116467727. / 12224457745473.
            self.b[4] = 1509636094297. / 3891594770934.

        self.stages = np.zeros((self.nstages, self.Ndof))

    def timestep(self, u0, dt):

        uend = u0
        for i in range(0, self.nstages):

            b = u0

            # Compute right hand side for this stage's implicit step
            for j in range(0, i):
                b = b + self.A[i, j] * dt * self.f(self.stages[j, :])

            # Implicit solve for current stage
            # if i==0:
            self.stages[i, :] = self.f_solve(b, dt * self.A[i, i], u0)
            # else:
            #  self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] , self.stages[i-1,:] )

            # Add contribution of current stage to final value
            uend = uend + self.b[i] * dt * self.f(self.stages[i, :])

        return uend

    #
    # Returns f(u) = c*u
    #
    def f(self, u):
        return self.problem.D_upwind.dot(u) + self.problem.M.dot(u)

    #
    # Solves (Id - alpha*c)*u = b for u
    #
    def f_solve(self, b, alpha, u0):
        cb = Callback()
        sol, info = gmres(self.problem.Id - alpha * (self.problem.D_upwind + self.problem.M), b, x0=u0,
                          tol=self.problem.params.gmres_tol_limit, restart=self.problem.params.gmres_restart,
                          maxiter=self.problem.params.gmres_maxiter, callback=cb)
        if alpha != 0.0:
            self.logger.add(cb.getcounter())
        return sol
