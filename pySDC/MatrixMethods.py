# coding=utf-8
import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
import scipy.linalg as la
import scipy.interpolate as intpl
from scipy.integrate import quad
from scipy.linalg import block_diag
# import matplotlib.pyplot as plt
# import matplotlib as mplt
# import pylab
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pySDC.tools.transfer_tools import to_sparse, to_dense
from pySDC.tools.plot_tools import matrix_plot
from pySDC.transfer_classes.transfer_between_meshes import time_mesh_to_time_mesh,composed_mesh_to_composed_mesh
import functools as ft
# Own collection of tools, require the imports from above. But first we will try to
# to use and enhance the tools given by pySDC
#from pint_matrix_tools import *


# We start with a bunch of helper functions and classes

class Bunch:
    """
    Create an object(Bunch) with some Attributes you initialize in the beginning.

    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def prepare_function(f):
    """
    This prepares a function for a simple use of numpy arrays to get a vector f(numpy.array) = numpy.array

    :param f: A function
    :return: Prepared function
    """
    return np.vectorize(f)


def extract_vector(f, u):
    """
    We expect a function which has arbitrary many arguments f(t,x,y,z, ...),
    the convention is that the first is argument is always time

    :param f: a function f(t,x,y,z,...)
    :param u: u is a list of numpy.arrays for each dimension containing the values one considers to evaluate
    :return: vector with the evaluated rhs
    """
    if len(u) is 1:
        return prepare_function(f)(u[0])
    else:
        vect_list = []
        next_u = u[1:]
        for x in u[0]:
            next_f = ft.partial(f, x)
            vect_list.append(extract_vector(next_f, next_u))
        return np.concatenate(vect_list)


def distributeToFirst(v, N):
    """
    Distribute to first, fill up with zeros
    :param v: numpy vector
    :param N: number of times shape is used
    :return: V=(v,0,...,0)
    """
    z = np.zeros(v.shape)
    vlist = [v]+[z]*(N-1)
    return np.concatenate(vlist)


def distributeToAll(v, N):
    """
    Distribute to all
    :param v: numpy vector
    :param N: number of times shape is repeated
    :return: V=(v,v,...,v)
    """
    vlist = [v]*(N)
    return np.concatenate(vlist)


def transform_to_unit_interval(x,t_l,t_r):
    return (x-t_l)/(t_r-t_l)

def sparse_inv(P,v):
    """
    Uses sparse solver to compute P^-1 * v
    :param P: sparse_matrix
    :param v: dense vector
    :return:
    """
    return spla.spsolve(P,v)


# now some classes building different classes of Linear Iterative Solver, using sparse matrices

class IterativeSolver(object):
    """ The basic Iterative Solver class,
        several steps of the iterative solver are called sweeps"""
    def __init__(self, P, M, c, sparse_format="array"):
        assert P.shape == M.shape and c.shape[0] == M.shape[0], \
            "Matrix P and matrix M don't fit.\n \tM_shape:" + str(M.shape) + " P_shape:" \
            + str(P.shape) + " c_shape:" + str(c.shape)
        self.format = sparse_format
        self.P = to_sparse(P, sparse_format)
        self.M = to_sparse(M, sparse_format)
        self.c = to_sparse(c, sparse_format)

    def invert_P(self):
        return la.inv(self.P)

    @property
    def P_inv(self):
        if self.format is "array":
            return self.invert_P()
        else:
            # define a function to compute P_inv.dot()
            return Bunch(dot=self.lin_solve)

    def step(self, U_last):
        # return U_last + np.dot(self.P_inv, (self.c - np.dot(self.M,U_last)))
        return U_last + self.P_inv.dot(self.c - self.M.dot(U_last))

    def sweep(self, U_0, k):
        if k == 0:
            return U_0
        U_last = np.copy(U_0)
        for i in range(k):
            U_last[:] = self.step(U_last)[:]
        return U_last

    @property
    def it_matrix(self):
        if self.format is "array":
            return np.eye(self.P.shape[0])-np.dot(self.P_inv, self.M)
        else:
            func = lambda v: v - self.P_inv.dot(self.M.dot(v))
            it_matrix = Bunch(dot=func)
            return it_matrix

    def lin_solve(self, v):
        return sparse_inv(self.P, v)


class SDCIterativeSolver(IterativeSolver):
    """
    Takes a level and returns an matrix SDC sweeper
    for this level.
    """

    def __init__(self, level, c, sparse_format="array"):
    #         M_list = map(lambda x, y, dt: sprs.eye(x.sweep.coll.num_nodes*y.system_matrix.shape[0], format=sparse_format)
    #                               -dt * sprs.kron(x.sweep.coll.Qmat[1:, 1:], y.system_matrix, format=sparse_format),
    #              fine_levels, problems_fine, dt_list)
    # P_f_list = map(lambda x, y, dt: sprs.eye(x.sweep.coll.num_nodes*y.system_matrix.shape[0], format=sparse_format)
    #                                 -dt*sprs.kron(x.sweep.coll.QDmat, y.system_matrix, format=sparse_format),
    #              fine_levels, problems_fine, dt_list)
        self.sparse_format = sparse_format
        # explicit part of the system matrix
        self.A_E = level.prob.A_E
        # implicit part of the system matrix
        self.A_I = level.prob.A_I
        # Q matrix
        self.Q = level.sweep.coll.Qmat[1:, 1:]
        self.QD = level.sweep.coll.QDmat
        # M matrix
        A = self.A_E + self.A_I
        dt = level.dt
        E = to_dense(sprs.dia_matrix((np.ones(self.Q.shape[0]-1), -1), shape=self.Q.shape))
        M = sprs.eye(self.A_E.shape[0]*self.Q.shape[0], format=sparse_format) \
            - sprs.kron(dt*self.Q, A, format=sparse_format)
        # print type(self.QD.dot(E))
        # print to_dense(self.QD.dot(E))
        P = sprs.eye(self.A_E.shape[0]*self.QD.shape[0], format=sparse_format) \
            - sprs.kron(dt*self.QD, self.A_I, format=sparse_format) \
            - sprs.kron(dt*self.QD.dot(E), self.A_E, format=sparse_format)
        if c is None:
            c = np.zeros(P.shape[0])

        super(SDCIterativeSolver, self).__init__(P, M, c, sparse_format=sparse_format)


class BlockDiagonalIterativeSolver(IterativeSolver):
    """
    Takes a list of iterative solvers and put them together into a Block Diagonal solver.
    Needs a function to tell how the different problems are connected, in order to
    describe how the different problems are connected.
    """

    def __init__(self, IterativeSolverList, connect_function=None, sparse_format="array"):
        self.it_list = IterativeSolverList
        self.sparse_format = sparse_format

        P_list = []
        M_list = []
        c_list = []
        shape_list = []

        for it_solv in IterativeSolverList:
            P_list.append(to_sparse(it_solv.P, sparse_format))
            M_list.append(to_sparse(it_solv.M, sparse_format))
            c_list.append(it_solv.c)
            shape_list.append(it_solv.P.shape[0])

        self._P_list = P_list
        self.__split_points = np.cumsum(shape_list)[:-1]
        self.P = sprs.block_diag(P_list, format=self.sparse_format)
        if connect_function is None:
            def connect_function(M_list, c_list, sparse_format):
                return sprs.block_diag(M_list, format=self.sparse_format), np.concatenate(c_list)

        self.M, self.c = connect_function(M_list, c_list, sparse_format)
        # self.P_inv = sprs.block_diag(*P_inv_list)

    @property
    def P_inv(self):
        """
        We build a function which splits a vector and solves each part of it with the solvers, given by the Iterative
        Solver list.
        :return: object with the attribute dot, which
        """
        if self.sparse_format is "array":
            return sprs.block_diag(map(lambda x: to_dense(x.P_inv), self.it_list), "array")
        else:
            def func(v):
                solutions = map(lambda f, x: f.P_inv.dot(x), self.it_list, np.split(v, self.__split_points))
                return np.concatenate(solutions)

            return Bunch(dot=func)

    def split(self, v):
        """ Splits a vector according to the block sizes"""
        return np.split(v, self.__split_points)

# TODO: this is just a copy from LinearPFASST.ipynb has to be adjusted to be used for sparse matrices
class MultiSolverIterativeSolver(IterativeSolver):
    """
    Takes a list of Iterative Solvers, which are all applied on the same vector for the
    same system-matrix M
    """
    def __init__(self, IterativeSolverList):
        self.M = IterativeSolverList[0].M
        self.c = IterativeSolverList[0].c
        self.P_inv = IterativeSolverList[0].P_inv

        for it_solv in IterativeSolverList[1:]:
            self.P_inv[:] = self.combine_two_P(self.P_inv, it_solv.P_inv)
            self.c[:] = self.combine_two_c(self.c,it_solv.c,self.P_inv,it_solv.P_inv)

        self.P = la.inv(self.P_inv)

    def combine_two_P(self ,P_1 ,P_2):
        return P_1 + P_2 - np.dot(P_2,np.dot(self.M,P_1))

    def combine_two_c(self ,c_1 ,c_2 ,P ,P_2):
        return c_1 + np.dot(np.dot(la.inv(P),P_2),c_2-c_1)


# TODO: See if this becomes useful someday
class MultiDimensionalIterativeSolver(IterativeSolver):
    """
    Assume you have two directions, consisting of several dimensions.
    E.g. A kron B as a Problem, then we maybe use A_delta kron B_delta as
        a preconditioner and we want different solvers to compute
                A_delta^-1 * v
        and     B_delta^-1 * v
    """

    def __init__(self, iterative_solver_list):
        pass


class MultiStepIterativeSolver(IterativeSolver):
    """ A Iterative solver which does one iteration on one interval, passes the
        result to the next block and so on.

        Examples:
        With this class and the right parameters it is possible to form for example a serial blockSDC solver.
        Using

    """
    def __init__(self, iterative_solver_list, nodes_on_unit_list, N_x_list=0, sparse_format="array"):
        """
        Construct the matrices which are needed for the IterativeSolver class

        :param iterative_solver_list: A list of IterativeSolver objects
        :param nodes_on_unit_list: a list containing the nodes are used on each subinterval
        :param N_x_list: list of the number of points in the spatial space
        :param sparse_format: tells the sparse format which should be used for the matrices
        :return: itself
        """
        self.n_blocks = len(iterative_solver_list)
        self.sparse_format = sparse_format
        self.it_list = iterative_solver_list

        self.M = self.distributeOperatorsToFullInterval(map(lambda x: to_sparse(x.M, sparse_format),
                                                            iterative_solver_list),
                                                        nodes_on_unit_list, N_x_list)
        self.P = self.distributeOperatorsToFullInterval(map(lambda x: to_sparse(x.P, sparse_format),
                                                            iterative_solver_list),
                                                        nodes_on_unit_list, N_x_list)
        self.c = np.concatenate(map(lambda x: to_sparse(x.c, sparse_format), iterative_solver_list))

        self.N_list = self.get_matrixN_list(map(lambda x: x.M, iterative_solver_list),
                                            nodes_on_unit_list, N_x_list)

        shape_list = map(lambda x: x.P.shape[0], iterative_solver_list)
        self.__split_points = np.cumsum(shape_list)[:-1]
        self.block_shapes = map(lambda x: x.P.shape, iterative_solver_list)

    @staticmethod
    def matrixN(tau, rows=-1, last_value=1.0):
        n = tau.shape[0]
        if rows == -1:
            rows = n
        N = np.zeros((rows, n))
        # construct the lagrange polynomials
        circulating_one = np.asarray([1.0]+[0.0]*(n-1))
        lag_pol = []
        for i in range(n):
            lag_pol.append(intpl.lagrange(tau, np.roll(circulating_one, i)))
            N[:, i] = -np.ones(rows)*lag_pol[-1](last_value)
        return N


    def distributeOperatorsToFullInterval(self, operator_list, tau_list, N_x_list=0):
        if N_x_list is 0:
            N_x = np.ones(len(tau_list))
        else:
            N_x = N_x_list
        # n = sum(map(lambda x: x.shape[0], operator_list))
        M = sprs.block_diag(operator_list, format=self.sparse_format)
        n_x = 0
        n_y = operator_list[0].shape[1]
        for i in range(len(operator_list)-1):
            x_plus = operator_list[i].shape[0]
            y_plus = operator_list[i+1].shape[1]
            M[n_y:n_y+y_plus, n_x:n_x + x_plus] = sprs.kron(self.matrixN(tau_list[i], y_plus/N_x[i+1]), np.eye(N_x[i])
                                                            , format=self.sparse_format)
            n_x = n_x + x_plus
            n_y = n_y + y_plus
        return M


    def get_matrixN_list(self, operator_list, tau_list, N_x_list=0):
        if N_x_list is 0:
            N_x = np.ones(len(tau_list))
        else:
            N_x = N_x_list
        # n = sum(map(lambda x: x.shape[0], operator_list))
        M = []
        for i in range(len(operator_list)-1):
            y_plus = operator_list[i+1].shape[1]
            M.append(np.kron(self.matrixN(tau_list[i], y_plus/N_x[i+1]), np.eye(N_x[i])))
        return M

    def split(self, v):
        """ Splits a vector according to the block sizes"""
        return np.split(v, self.__split_points)

    @property
    def P_inv(self):
        # the inverse is computed by solving and passing information
        def solver(v):
            v_split = self.split(v)
            v_solution = []
            v_transfer = np.zeros(v_split[0].shape)
            for i in range(self.n_blocks):
                v_solution.append(self.it_list[i].P_inv.dot(v_split[i]-v_transfer))
                if i < self.n_blocks-1:
                    v_transfer = self.N_list[i].dot(v_solution[-1])
            return np.concatenate(v_solution)

        if self.sparse_format is "array":
            return la.inv(self.P)
        else:
            return Bunch(dot=solver)

#TODO: I am not happy with the structure of this function
class TransferredMultiStepIterativeSolver(MultiStepIterativeSolver):
    """
    Like the multi-step solver expect the difference that the pre-conditioner is
    solved on a coarser level and then transferred back to the fine level.
    this class is needed for the LinearPFASST.
    """
    def __init__(self, iterative_solver_list, nodes_on_unit_list, transfer_list,
                 P_c_list, N_x_list_fine=0, N_x_list_coarse=0, sparse_format="array"):

        super(TransferredMultiStepIterativeSolver, self).__init__(iterative_solver_list, nodes_on_unit_list,
                                                                  N_x_list_fine, sparse_format)
        self.transfer_list = transfer_list
        self.P_c_list = map(lambda x: to_sparse(x, sparse_format), P_c_list)
        self.P_c_inv_list = map(lambda x: la.inv(x), self.P_c_list)
        # N_x_list

        self.P = self.distributeOperatorsToFullInterval(map(lambda x, y: y.Pspace.dot(x.dot(y.Rspace)),
                                                            P_c_list, transfer_list),
                                                        nodes_on_unit_list, N_x_list_fine)
        self.__nodes_on_unit = nodes_on_unit_list
        self.__N_x_list_fine = N_x_list_fine
        self.__N_x_list_coarse = N_x_list_coarse
        self.__transfer_list = transfer_list

    @property
    def P_inv(self):

        if self.sparse_format is "array":
            P_c_inv = map(lambda x, y: x.Pspace.dot(y.dot(x.Rspace)), self.__transfer_list, self.P_c_inv_list)
            return self.distributeOperatorsToFullInterval(P_c_inv, self.__nodes_on_unit, self.__N_x_list_fine)
        else:
            def solver(v):
                v_split = self.split(v)
                v_solution = []
                v_transfer = np.zeros(v_split[0].shape)
                for i in range(self.n_blocks):
                    # T_gf P_inf T
                    v_solution.append(
                        self.transfer_list.Pspace.dot(
                            self.P_c_inv[i].dot(
                                self.transfer_list[i].Rspace.dot(v_split[i]-v_transfer))))

                    if i < self.n_blocks-1:
                        v_transfer = self.N_list[i].dot(v_solution[-1])
                return np.concatenate(v_solution)

            return Bunch(dot=solver)



class LinearPFASST(IterativeSolver):
    """ LinearPFASST with just two Level
        The CoarseSolver should be a serial BlockSDC_Sweeper and
        the FineSolver a parallel BlockDiagonalIterativeSolver.

        Note that the coarseSolver is not really used but that a new solver is reconstructed
        implicitly through the transferoperators.
    """
    def __init__(self, multistep_iterative_solver, block_iterative_solver, transfer_list, sparse_format="array"):
        # gather some important values from step_list
        self.__ms_its = multistep_iterative_solver
        self.__bl_its = block_iterative_solver
        self.sparse_format = sparse_format

        self.N_t = multistep_iterative_solver.n_blocks       # number of subintervals
        self.P_c_inv = multistep_iterative_solver.P_inv
        self.P_f_inv = block_iterative_solver.P_inv
        self.P_c = multistep_iterative_solver.P
        self.P_f = block_iterative_solver.P
        self.M = multistep_iterative_solver.M
        self.c = multistep_iterative_solver.c


        # self.P_c_inv = np.dot(self.T_gf,np.dot(CoarseSolver.P_inv,self.T_fg))
        # self.P_f_inv = FineSolver.P_inv
        # self.P_c = CoarseSolver.P
        # self.P_c_transfered = np.dot(self.T_gf,np.dot(self.P_c,self.T_fg))
        # self.P_f = FineSolver.P
        # self.M = FineSolver.M
        # self.c = FineSolver.c
        # # self.P = la.inv(self.P_inv)
        self.U_half = np.zeros(self.M.shape[0])

    @property
    def P(self):
        return self.combine_two_P(self.P_c_inv, self.P_f_inv)

    @property
    def multi_step_solver(self):
        return self.__ms_its

    @property
    def block_diag_solver(self):
        return self.__bl_its

    @property
    def P_inv(self):
        """
        The preconditioner for the coarse level including coarse grid correction and
        communication between subintervals
        :return:
        """
        if self.sparse_format is "array":
            return self.combine_two_P(self.P_c_inv, self.P_f_inv)
        else:
            def solver(v):
                return self.P_c_inv.dot(v) + self.P_f_inv.dot(v) - self.P_f_inv.dot(self.M.dot(self.P_c_inv.dot(v)))

            return Bunch(dot=solver)

    def combine_two_P(self ,P_1 ,P_2):
        return P_1 + P_2 - np.dot(P_2,np.dot(self.M,P_1))

    def combine_two_c(self ,c_1 ,c_2 ,P ,P_2):
        return c_1 + np.dot(np.dot(la.inv(P),P_2),c_2-c_1)

    def step(self, U_last):
        # print(self.M.dot(U_last).T.shape, self.c.shape, self.P_c_inv.shape,
        #       self.P_c_inv.dot((self.c.reshape(1,-1) - self.M.dot(U_last).reshape(1,-1))).shape,
        #       U_last.shape)
        self.U_half[:] = U_last + self.P_c_inv.dot((self.c - self.M.dot(U_last).T)).T
        return self.U_half + self.P_f_inv.dot(self.c - self.M.dot(self.U_half))

    def sweep(self,U_0,k):
        if k==0:
            return U_0
        U_last = np.copy(U_0)
        for i in range(k):
            U_last[:] = self.step(U_last)[:]
        return U_last

    @property
    def it_matrix(self):
        """
        :return: iteration matrix of the LinPFASST
        """
        M = to_dense(self.M)
        P_f_inv = to_dense(self.P_f_inv)
        P_c_inv = to_dense(self.P_c_inv)
        I = np.eye(M.shape[0])
        # print(map(type,[M,self.P_f_inv,P_c_inv,I]))
        return np.dot(I - np.dot(P_f_inv, M), I - np.dot(P_c_inv, M))

    def check_condition_numbers(self, p=2):
        K_M = np.linalg.cond(self.M, p=p)
        K_P = np.linalg.cond(self.P, p=p)
        # K_P_c = np.linalg.cond(self.P_c, p=np.inf)
        K_P_f = np.linalg.cond(self.P_f, p=p)
        norm_b = np.linalg.norm(self.c, ord=p)
        print "K(M) \t : \t", K_M
        print "K(P) \t : \t", K_P
        # print "K(P_c) \t : \t", K_P_c
        print "K(P_f) \t : \t", K_P_f
        print "norm(c) \t : \t", norm_b
        print "rock_bottom \t : \t", norm_b/K_M * 9.9E-5
        print "top_heaven\t : \t", norm_b*K_M * 9.9E-5

        # now do it for all subblocks to compare with
        K_M_sum = 0
        for it in self.__ms_its.it_list:
            K_M_sum += np.linalg.cond(it.M, p=p)
            print "\tK(M_sub) \t : \t", np.linalg.cond(it.M, p=p)
            print "\tK(P_sub) \t : \t", np.linalg.cond(it.P, p=p)
        print "sum K_M_sub \t : \t", K_M_sum

    def spectral_radius(self, ka=1,tolerance=1e-7):
        # use iterative scheme to get the first couple of eigenvalues faster
        # print "real spectralrad:", np.max(np.abs(np.linalg.eigvals(self.it_matrix)))
        # some randome experiments showed that tolerance of at least 1e-6, but one needs to compute
        # more than one eigenvalue at once, because else it shows instabilities
        return np.abs(spla.eigs(self.it_matrix, k=ka, which='LM', tol=tolerance, return_eigenvectors=False))
        # return np.max(np.abs(np.linalg.eigvals(self.it_matrix)))

    def parallel_efficiency(self, c_c_tuple, c_f_tuple, c_t_tuple, use_dimension_of_A=True):
        """ compute the parallel efficiency of the LinearPFASST setup, given the cost of evaluating and solving
            the problem in the spatial space
            c_c - coarse sweep costs, c_f - fine sweep costs
                c_(c/f)_tuple[0] - cost of evaluation (per dof - only if use_dimension_of_A == True) for the implicit part
                c_(c/f)_tuple[1] - cost of evaluation (per dof - only if use_dimension_of_A == True) for the explicit part
                c_(c/f)_tuple[2] - cost of solving    (per dof - only if use_dimension_of_A == True)
            c_t - transfer/communication costs
                c_t_tuple[0] - interpolation cost per number of coarse nodes
                c_t_tuple[1] - restriction cost per number of fine node
                c_t_tuple[2] - communication cost in general per transfer
        """

        # number of intervals
        N_t = self.N_t

        # cost of a serial sweep only on the fine level
        C_s = 0
        for a_i, a_e, n_nodes in zip(self.FineSolver.A_I_list, self.FineSolver.A_E_list, self.FineSolver.N_nodes):
            if use_dimension_of_A:
                C_s += n_nodes * (2*c_f_tuple[0]*a_i.shape[0]+(2*c_f_tuple[1]+c_f_tuple[2])*a_e.shape[0])
            else:
                C_s += n_nodes * (2*c_f_tuple[0]+2*c_f_tuple[1]+c_f_tuple[2])

        # cost of a LinearPFASST step
        # the serial part is also part of the parallel part, but may be distributed on N processors
        C_p = C_s/N_t
        #  the communication, restriction and interpolation costs are added
        C_p += (N_t-1)*c_t_tuple[2]
        for a_i, a_e, n_nodes_f, n_nodes_c in zip(self.CoarseSolver.A_I_list, self.CoarseSolver.A_E_list, \
                                                  self.FineSolver.N_nodes, self.CoarseSolver.N_nodes):
            if use_dimension_of_A:
                C_p += n_nodes_c * (2*c_c_tuple[0]*a_i.shape[0]+(2*c_c_tuple[1]+c_c_tuple[2])*a_e.shape[0])
                C_p += n_nodes_f * c_t_tuple[1] + n_nodes_c * c_t_tuple[0]
            else:
                C_p += n_nodes_c * (2*c_c_tuple[0]+(2*c_c_tuple[1]+c_c_tuple[2]))
                C_p += c_t_tuple[1] + c_t_tuple[0]

        # The parallel efficiency can be measured by
        #  P = N_t (C_s log(lambda_p))/((C_s+C_p/N) log(lambda_s))
        lambda_p = self.spectral_radius()
        # Hier bin ich mir nicht sicher ob das richtig ist
        lambda_s = self.FineSolver.spectral_radius_serial()

        print("C_p \t\t C_s \t\t lam_p \t\t lam_s \n",C_p," \t\t ",C_s," \t\t ", lambda_p, " \t\t ", lambda_s)
        print("Parallel efficieny:",  (C_s*np.log(lambda_p))/(C_p*np.log(lambda_s)))
        return (C_s*np.log(lambda_p))/(C_p*np.log(lambda_s))


def generate_LinearPFASST(step_list, transfer_list, u_0, **kwargs):

    if 'sparse_format' in kwargs:
        sparse_format = kwargs['sparse_format']
    else:
        sparse_format = "array"

    if 't_0' in kwargs:
        t_0 = kwargs['t_0']
    else:
        t_0 = 0

    # generate MultiStepSolver
    # put them together into LinearPFASST and return the object
    # first we need the system matrix, e.g. Q kron A
    fine_levels = map(lambda x: x.levels[0], step_list)
    coarse_levels = map(lambda x: x.levels[-1], step_list)
    problems_fine = map(lambda x: x.prob, fine_levels)
    problems_coarse = map(lambda x: x.prob, coarse_levels)
    dt_list = map(lambda x: x.status.dt, step_list)

    # M_list = map(lambda x, y, dt: sprs.eye(x.sweep.coll.num_nodes*y.system_matrix.shape[0], format=sparse_format)
    #                               -dt * sprs.kron(x.sweep.coll.Qmat[1:, 1:], y.system_matrix, format=sparse_format),
    #              fine_levels, problems_fine, dt_list)
    # P_f_list = map(lambda x, y, dt: sprs.eye(x.sweep.coll.num_nodes*y.system_matrix.shape[0], format=sparse_format)
    #                                 -dt*sprs.kron(x.sweep.coll.QDmat, y.system_matrix, format=sparse_format),
    #              fine_levels, problems_fine, dt_list)
    c_0 = np.kron(np.ones(fine_levels[0].sweep.coll.num_nodes), u_0)
    # for the comparison use the SDC IterativeSolver class and adjust the c_list to take
    # in account forcing terms
    # das muss anders gehen

    # compute the quadrature of the forcing term

    all_nodes_split = np.split(get_all_nodes(step_list, t_0), len(step_list))
    # values of the forcing term
    f_values = map(lambda prob, nodes: prob.force_term(nodes), problems_fine, all_nodes_split)
    # quadrature of the f_values
    Qf_values = map(lambda lvl, f, dt: dt*sprs.kron(lvl.sweep.coll.Qmat[1:, 1:], sprs.eye(lvl.prob.nvars),
                                             format=sparse_format).dot(f), fine_levels, f_values, dt_list)

    c_list = [c_0+Qf_values[0]] + Qf_values[1:]


    sdc_it_solver_list = map(lambda x, y: SDCIterativeSolver(x, y, sparse_format=sparse_format), fine_levels, c_list)

    # offset = int(not fine_levels[0].sweep.coll.left_is_node)

    # it_solver_list = [IterativeSolver(P_f_list[0], M_list[0], c_0)] \
    #                  + map(lambda P, M, x: IterativeSolver(P, M, np.zeros(P.shape[0])),
    #                        P_f_list[1:], M_list[1:], fine_levels[1:])
    # dadurch das c gegeben ist

    # it_solver_list = map(lambda P, M, c: IterativeSolver(P, M, c), P_f_list, M_list, c_list)
    it_solver_list = sdc_it_solver_list
    # print c_list
    # to test if the

    # construct the block solver
    block_solver = BlockDiagonalIterativeSolver(it_solver_list)

    # construct the multistep solver
    # pre-conditioner for the coarse level
    P_c_list = map(lambda x, y, dt:sprs.eye(x.sweep.coll.num_nodes*y.system_matrix.shape[0], format=sparse_format)
                                   - dt*sprs.kron(x.sweep.coll.QDmat, y.system_matrix),
                   coarse_levels, problems_coarse, dt_list)
    # nodes on unit interval
    nodes_on_unit = map(lambda x: x.sweep.coll.nodes, fine_levels)
    N_x_list_fine = map(lambda x: x.nvars, problems_fine)
    N_x_list_coarse = map(lambda x: x.nvars, problems_coarse)
    multi_step_solver = TransferredMultiStepIterativeSolver(it_solver_list, nodes_on_unit,
                                                            transfer_list, P_c_list, N_x_list_fine, N_x_list_coarse)

    return LinearPFASST(multi_step_solver, block_solver, transfer_list)


def generate_transfer_list(step_list, transfer_class, **kwargs):
    """
    This takes the generated step list and constructs a list
    of Transfer-objects, for each subinterval. This is needed to
    construct the LinearPFASST object.
    :param step_list: List of steps
    :param transfer_class: transferclass which is used for all steps
    :return:
    """
    # Das Problem is das in PySDC die Transfer nur in Raumrichtung stattfinden, was aber nötig ist
    # für eine allgemeinere darstellung ist der nutzen von ZeitRaum, die frage ist wo ich es reinpacke
    # am besten hierhin weil es eh nicht fertig ist.

    # the transfer object list just for space
    space_transfer_list = map(lambda x, int_ord, restr_ord: transfer_class(x.levels[0], x.levels[-1],
                                                                           int_ord, restr_ord, kwargs['sparse_format']),
                              step_list, kwargs['interpolation_order'], kwargs['restriction_order'])

    # the transfer list for the time component are extracted not from the level but more from the sweeper which is used
    # time_transfer_list = map(lambda )
    time_transfer_list = map(lambda x, int_ord, restr_ord: time_mesh_to_time_mesh(x.levels[0], x.levels[-1],
                                                                           int_ord, restr_ord, kwargs['sparse_format']),
                              step_list, kwargs['t_interpolation_order'], kwargs['t_restriction_order'])

    # return the composition of the two transfer classes, very hacky
    return map(lambda x, s_trans, t_trans: composed_mesh_to_composed_mesh(x.levels[0], x.levels[-1],
                                                                          [t_trans, s_trans], kwargs['sparse_format']),
               step_list, space_transfer_list, time_transfer_list)


def get_all_nodes(steps, t_0=0):
    dts = map(lambda x: x.status.dt, steps)
    offsets = np.cumsum([0.0]+dts)[:-1]
    return np.hstack(map(lambda x, dt, off: x.levels[0].sweep.coll.nodes*dt+off+t_0, steps, dts, offsets))


def run_linear_pfasst(lin_pfasst, u_0, kwargs):
    """
    Runs linear pfasst.
    :param lin_pfasst: The Linear PFASST object
    :param u_0: initial value
    :param kwargs: options
    :return: list of residuals, list_of_solution
    """
    run_type = kwargs['run_type']
    norm = kwargs['norm']
    # debug = kwargs['debug']
    if run_type is "max_it":
        # runs max_it iterations.
        max_it = kwargs['max_it']
        u = [u_0]
        res = [norm(lin_pfasst.M.dot(u_0) - lin_pfasst.c)]
        for i in range(max_it):
            u.append(lin_pfasst.step(u[-1]))
            res.append(norm(lin_pfasst.M.dot(u[-1])-lin_pfasst.c))
    elif run_type is "tolerance":
        # runs until a certain norm(residuum) is achieved
        tol = kwargs['tol']
        u = [u_0]
        res = [norm(lin_pfasst.M.dot(u_0) - lin_pfasst.c)]
        while res[-1] > tol and len(u) < 50:
            u.append(lin_pfasst.step(u[-1]))
            res.append(norm(lin_pfasst.M.dot(u[-1])-lin_pfasst.c))
    else:
        raise TypeError("This run type is undefined")


    return res, u

def run_multiple_linear_pfasst(steps, decomposition_of_time, **kwargs):
    """
    Generates linear_pfasst objects and transfer objects according to the decomposition of time.
    The grouping of the intervals in decomposition_of_time could be for example
        [[t_0,t_1,t_2],[t_2,t_3],[t_3,t_4,t_5,T_end]]
    it translates to 3 linear_pfasst objects, where the first has 3 blocks, the second just one and the last 3 blocks.
    """
    # as preparation we need a lists of parameter dicts, which we get from **kwargs
    # we start with the transfer parameter list
    num_procs = len(steps)
    if 'tparams_list' in kwargs:
        tparams_list = kwargs['tparams_list']
    else:
        tparams = {}
        tparams['finter'] = True
        tparams['sparse_format'] = "array"
        tparams['interpolation_order'] = [[3, 10]]*num_procs
        tparams['restriction_order'] = [[3, 10]]*num_procs
        tparams['interpolation_order'] = [[3, 10]]*num_procs
        tparams['restriction_order'] = [[3, 10]]*num_procs
        tparams['t_interpolation_order'] = [2]*num_procs
        tparams['t_restriction_order'] = [2]*num_procs
        tparams_list = [tparams]*len(decomposition_of_time)

    if 'run_type_list' in kwargs:
        run_type_list = kwargs['run_type_list']
    else:
        run_type_list = ['tolerance']*len(decomposition_of_time)



    # decompose steps accordingly
    numb_of_blocks = [(len(p) - 1) for p in decomposition_of_time]
    assert sum(numb_of_blocks) is len(steps)
    step_list = [steps[i:j] for i, j in zip(np.cumsum([0]+numb_of_blocks[:-1]), np.cumsum(numb_of_blocks))]

    # construct the first transfer_list and linear_pfasst object

class LFAForLinearPFASST:
    """ Takes a LinearPFASST object extracts the submatrices needed and performs a LFA by
        computing the eigenvalues of the different fouriersymbols.
        Additionally it compares the results with the eigenvalues itself.
        Note that the blocks must be identical.
    """
    def __init__(self, lin_pfasst_solver, steps, transfer_list, debug=False):
        self.debug = debug
        self.lin_pfasst_solver = lin_pfasst_solver
        # extracted from the time_space
        # self.nn_t_coarse = time_space.nn_coarse
        self.nn_t_coarse = steps[0].levels[-1].sweep.coll.num_nodes * steps[0].levels[-1].prob.nvars
        # self.nn_t_fine = time_space.nn_fine
        self.nn_t_fine = steps[0].levels[0].sweep.coll.num_nodes * steps[0].levels[0].prob.nvars
        # self.N_t = time_space.N_t
        self.N_t = len(steps)
        # extracted from the fine Solver
        pf1_col = lin_pfasst_solver.multi_step_solver.block_shapes[0][1]
        pf1_row = lin_pfasst_solver.multi_step_solver.block_shapes[0][0]
        pf2_row = lin_pfasst_solver.multi_step_solver.block_shapes[1][0]

        self.N_f = lin_pfasst_solver.M[pf1_row:pf1_row+pf2_row, 0:pf1_col]
        self.M_f = lin_pfasst_solver.M[0:pf1_row, 0:pf1_col]
        self.P_f = lin_pfasst_solver.P_f[0:pf1_row, 0:pf1_col]
        self.P_f_inv = lin_pfasst_solver.P_f_inv[0:pf1_row, 0:pf1_col]

        # extracted from the coarse Solver
        # generate SDCSolver for the coarse level
        coarse_level = steps[0].levels[-1]
        sdc_solver = SDCIterativeSolver(coarse_level, None)
        ms_solver = MultiStepIterativeSolver([sdc_solver]*2,
                                             [coarse_level.sweep.coll.nodes]*2,
                                             [coarse_level.prob.nvars]*2)

        pc1_col = ms_solver.block_shapes[0][1]
        pc1_row = ms_solver.block_shapes[0][0]
        pc2_row = ms_solver.block_shapes[1][0]

        self.N_c = ms_solver.M[pc1_row:pc1_row+pc2_row, 0:pc1_col]
        self.M_c = sdc_solver.M
        self.P_c = sdc_solver.P
        self.P_c_inv = sdc_solver.P_inv

        # extracted from the linPFASST class (TransferOperator)
        # first we may compute the fine and coarse space time
        self.nn_x_fine = pf1_col/self.nn_t_fine
        self.nn_x_coarse = pc1_col/self.nn_t_coarse
        # we get the first
        self.T_fg = transfer_list[0].Rspace
        self.T_gf = transfer_list[0].Pspace


    def M_f_symbol(self,theta):
        """ Symbol of the system matrix on a fine level"""
        return self.M_f - np.exp(-1.0j * theta)*self.N_f

    def M_c_symbol(self,theta):
        """ Symbol of the system matrix on a coarse level"""
        return self.M_c - np.exp(-1.0j * theta)*self.N_c

    def S_symbol(self,theta):
        """ Smoother Symbol """
        return np.eye(self.nn_t_fine)-np.dot(self.P_f_inv, self.M_f_symbol(theta))

    def B_c_symbol_inv(self,theta):
        """ Symbol of the coarse sdc matrix """
        return la.inv(self.P_c - np.exp(-1.0j * theta) * self.N_c)

    def K_symbol(self,theta):
        """ Symbol of the coarse grid correction """
        return np.eye(self.nn_t_fine) - \
                    np.dot(np.dot(self.T_gf,np.dot(self.B_c_symbol_inv(theta),self.T_fg)),self.M_f_symbol(theta))

    def two_grid_symbol(self, theta, nu=1):
        """ combine all symbols above to a two_grid cycle"""
        S = np.linalg.matrix_power(self.S_symbol(theta), nu)
        return np.dot(S,self.K_symbol(theta))

    def low_thetas(self):
        N_L = self.N_t
        low_frequencies=[]
        for k in range(1-N_L/2,N_L/2+1):
            if (k*np.pi*2/N_L > -np.pi/2 and k*np.pi*2/N_L < np.pi/2):
                low_frequencies.append(k*np.pi*2/N_L)
        return low_frequencies

    def high_thetas(self):
        N_L = self.N_t
        high_frequencies=[]
        for k in range(1-N_L/2,N_L/2+1):
            if not (k*np.pi*2 > -np.pi/2 and k*np.pi*2 < np.pi/2):
                high_frequencies.append(k*np.pi*2/N_L)
        return high_frequencies

    def asymptotic_conv_factor(self,nu=1):
        current_max = 0.0
        for theta in self.low_thetas():
            spec_rad = np.max(np.abs(la.eigvals(self.two_grid_symbol(theta,nu))))

            if self.debug:
                print "Theta:\t",theta
                print "SpecRad:\t",spec_rad

            if spec_rad > current_max:
                current_max = spec_rad

        return current_max
