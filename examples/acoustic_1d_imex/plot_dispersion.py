from pySDC import Level as lvl
from pySDC import Hooks as hookclass
from pySDC import CollocationClasses as collclass
from pySDC import Step as stepclass

from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order as imex
from standard_integrators import dirk, rk_imex
# for simplicity, import the scalar problem to generate Q matrices
from examples.fwsw.ProblemClass import swfw_scalar 
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import sympy

from pylab import rcParams
import matplotlib.pyplot as plt
from subprocess import call

def findomega(stab_fh):
  assert np.array_equal(np.shape(stab_fh),[2,2]), 'Not 2x2 matrix...'
  omega = sympy.Symbol('omega')
  func = (sympy.exp(-1j*omega)-stab_fh[0,0])*(sympy.exp(-1j*omega)-stab_fh[1,1])-stab_fh[0,1]*stab_fh[1,0]
  solsym = sympy.solve(func, omega)
  sol0 = complex(solsym[0])
  sol1 = complex(solsym[1])
  if sol0.real>=0:
    sol = sol0
  elif sol1.real>=0:
    sol = sol1
  else:
    print "Two roots with real part of same sign..."
  return sol

if __name__ == "__main__":

    pparams = {}
    # the following are not used in the computation
    pparams['lambda_s'] = np.array([0.0])
    pparams['lambda_f'] = np.array([0.0])
    pparams['u0'] = 1.0
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLegendre
    swparams['num_nodes'] = 3
    K = 4
    dirk_order = K
    
    c_speed = 1.0
    U_speed = 0.05
    
    #
    # ...this is functionality copied from test_imexsweeper. Ideally, it should be available in one place.
    #
    step = stepclass.step(params={})
    L = lvl.level(problem_class=swfw_scalar, problem_params=pparams, dtype_u=mesh, dtype_f=rhs_imex_mesh, sweeper_class=imex, sweeper_params=swparams, level_params={}, hook_class=hookclass.hooks, id="stability")
    step.register_level(L)
    step.status.dt   = 1.0 # can't use different value
    step.status.time = 0.0
    u0 = step.levels[0].prob.u_exact(step.status.time)
    step.init_step(u0)
    nnodes  = step.levels[0].sweep.coll.num_nodes
    level   = step.levels[0]
    problem = level.prob
  
    QE = level.sweep.QE[1:,1:]
    QI = level.sweep.QI[1:,1:]
    Q  = level.sweep.coll.Qmat[1:,1:]
    Nsamples = 30
    k_vec = np.linspace(0, np.pi, Nsamples+1, endpoint=False)
    k_vec = k_vec[1:]
    phase = np.zeros((3,Nsamples))
    amp_factor = np.zeros((3,Nsamples))
    for i in range(0,np.size(k_vec)):
      Cs = -1j*k_vec[i]*np.array([[0.0, c_speed],[c_speed, 0.0]], dtype='complex')
      Uadv = -1j*k_vec[i]*np.array([[U_speed, 0.0], [0.0, U_speed]], dtype='complex')

      LHS = np.eye(2*nnodes) - step.status.dt*( np.kron(QI,Cs) + np.kron(QE,Uadv) )
      RHS = step.status.dt*( np.kron(Q, Uadv+Cs) - np.kron(QI,Cs) - np.kron(QE,Uadv) )

      LHSinv = np.linalg.inv(LHS)
      Mat_sweep = np.linalg.matrix_power(LHSinv.dot(RHS), K)
      for k in range(0,K):
        Mat_sweep = Mat_sweep + np.linalg.matrix_power(LHSinv.dot(RHS),k).dot(LHSinv)
      ## 
      # ---> The update formula for this case need verification!!
      update = step.status.dt*np.kron( level.sweep.coll.weights, Uadv+Cs )
      
      y1 = np.array([1,0], dtype='complex')
      y2 = np.array([0,1], dtype='complex')
      e1 = np.kron( np.ones(nnodes), y1 )
      stab_fh_1      = y1 + update.dot( Mat_sweep.dot(e1) )
      e2 = np.kron( np.ones(nnodes), y2 )
      stab_fh_2      = y2 + update.dot(Mat_sweep.dot(e2))
      stab_sdc = np.column_stack((stab_fh_1, stab_fh_2))
      
      # Stability function of backward Euler is 1/(1-z); system is y' = (Cs+Uadv)*y
      #stab_ie = np.linalg.inv( np.eye(2) - step.status.dt*(Cs+Uadv) )

      # For testing, insert exact stability function exp(-dt*i*k*(Cs+Uadv)
      #stab_fh = la.expm(Cs+Uadv)
      
      dirkts = dirk(Cs+Uadv, dirk_order)
      stab_fh1 = dirkts.timestep(y1, 1.0)
      stab_fh2 = dirkts.timestep(y2, 1.0)
      stab_dirk = np.column_stack((stab_fh1, stab_fh2))

      rkimex = rk_imex(M_fast = Cs, M_slow = Uadv, order = K)
      stab_fh1 = rkimex.timestep(y1, 1.0)
      stab_fh2 = rkimex.timestep(y2, 1.0)
      stab_rk_imex = np.column_stack((stab_fh1, stab_fh2))

      sol_sdc = findomega(stab_sdc)
      sol_dirk = findomega(stab_dirk)
      sol_rk_imex = findomega(stab_rk_imex)
      
      # Now solve for discrete phase 
      phase[0,i]      = sol_sdc.real/k_vec[i]
      amp_factor[0,i] = np.exp(sol_sdc.imag)
      phase[1,i]      = sol_dirk.real/k_vec[i]
      amp_factor[1,i] = np.exp(sol_dirk.imag)
      phase[2,i]      = sol_rk_imex.real/k_vec[i]
      amp_factor[2,i] = np.exp(sol_rk_imex.imag)

    ###
    rcParams['figure.figsize'] = 1.5, 1.5
    fs = 8
    fig  = plt.figure()
    plt.plot(k_vec, (U_speed+c_speed)+np.zeros(np.size(k_vec)), '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, phase[1,:], '-', color='g', linewidth=1.5, label='DIRK('+str(dirkts.order)+')')
    plt.plot(k_vec, phase[2,:], '-', color='r', linewidth=1.5, label='RK-IMEX('+str(rkimex.order)+')')
    plt.plot(k_vec, phase[0,:], '-', color='b', linewidth=1.5, label='SDC('+str(K)+')')
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Phase speed', fontsize=fs, labelpad=0.5)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([0.0, 1.1*(U_speed+c_speed)])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    filename = 'sdc-fwsw-disprel-phase-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

    fig  = plt.figure()
    plt.plot(k_vec, 1.0+np.zeros(np.size(k_vec)), '--', color='k', linewidth=1.5, label='Exact')
    plt.plot(k_vec, amp_factor[1,:], '-', color='g', linewidth=1.5, label='DIRK('+str(dirkts.order)+')')
    plt.plot(k_vec, amp_factor[2,:], '-', color='r', linewidth=1.5, label='RK-IMEX('+str(rkimex.order)+')')
    plt.plot(k_vec, amp_factor[0,:], '-', color='b', linewidth=1.5, label='SDC('+str(K)+')')
    plt.xlabel('Wave number', fontsize=fs, labelpad=0.25)
    plt.ylabel('Amplification factor', fontsize=fs, labelpad=0.5)
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.xlim([k_vec[0], k_vec[-1:]])
    plt.ylim([k_vec[0], k_vec[-1:]])
    plt.legend(loc='lower left', fontsize=fs, prop={'size':fs})
    plt.gca().set_ylim([0.0, 1.1])
    plt.xticks([0, 1, 2, 3], fontsize=fs)
    #plt.show()
    filename = 'sdc-fwsw-disprel-ampfac-K'+str(K)+'-M'+str(swparams['num_nodes'])+'.pdf'
    plt.gcf().savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

