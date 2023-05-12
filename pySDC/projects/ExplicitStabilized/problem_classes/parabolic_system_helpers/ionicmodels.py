import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from pySDC.core.Errors import ParameterError

class IonicModel:
    def __init__(self):
        #defined in subclasses
        self.size = None
        #defined later
        self.domain = None 

    def set_domain(self,domain):
        self.domain = domain
    
    def f(self,uh):
        raise Exception('Implement f')
    
    def f_stiff(self,uh):
        raise Exception('Implement f_stiff')
    
    def u_stiff_coeffs(self,uh):
        raise Exception('Implement u_exp_coeffs')
    
    def f_nonstiff(self,uh):
        raise Exception('Implement f_nonstiff')
    
    def f_expl(self,uh):
        raise Exception('Implement f_expl')
    
    def u_exp_coeffs(self,uh):
        raise Exception('Implement u_exp_coeffs')
    
    def initial_values(self):
        raise Exception('Implement initial_values')
    
class HodgkinHuxley(IonicModel):
    def __init__(self):
        super(HodgkinHuxley,self).__init__()

        self.size = 4

        self.gNa = 120
        self.gK = 36.
        self.gL = 0.3
        self.vNa = 40.
        self.vK = -87.
        self.vL = -64.387

        self.V0 = -75.
        self.m0 = 0.05
        self.n0 = 0.317
        self.h0 = 0.595

        self.f_stiff_args = [0,1]

    def f(self,uh):
        V, m, n, h = uh.values.split()
        
        am = -0.1*(V+50.)/(ufl.exp(-(V+50.)/10.)-1.)
        an = -0.01*(V+65.)/(ufl.exp(-(V+65.)/10.)-1.)
        ah = 0.07*ufl.exp(-(V+75)/20.)
        bm = 4.0*ufl.exp(-(V+75)/18.)
        bn = 0.125*ufl.exp(-(V+75)/80.)
        bh = 1.0/(ufl.exp(-(V+45.)/10.)+1.)
        
        ydot = [None]*self.size
        
        ydot[0] = -(self.gNa*m**3*h*(V-self.vNa)+self.gK*n**4*(V-self.vK)+self.gL*(V-self.vL))
        ydot[1] = am*(1.-m)-bm*m
        ydot[2] = an*(1.-n)-bn*n
        ydot[3] = ah*(1.-h)-bh*h

        return ydot
    
    def f_stiff(self,uh):
        V, m, n, h = uh.values.split()
        
        am = -0.1*(V+50.)/(ufl.exp(-(V+50.)/10.)-1.)
        bm = 4.0*ufl.exp(-(V+75)/18.)
        
        ydot = [None]*self.size
        
        ydot[1] = am*(1.-m)-bm*m

        return ydot
    
    def f_nonstiff(self,uh):
        V, m, n, h = uh.values.split()
        
        an = -0.01*(V+65.)/(ufl.exp(-(V+65.)/10.)-1.)
        ah = 0.07*ufl.exp(-(V+75)/20.)
        bn = 0.125*ufl.exp(-(V+75)/80.)
        bh = 1.0/(ufl.exp(-(V+45.)/10.)+1.)
        
        ydot = [None]*self.size
        
        ydot[0] = -(self.gNa*m**3*h*(V-self.vNa)+self.gK*n**4*(V-self.vK)+self.gL*(V-self.vL))
        ydot[2] = an*(1.-n)-bn*n
        ydot[3] = ah*(1.-h)-bh*h

        return ydot

    def f_expl(self, uh):
        V, m, n, h = uh.values.split()

        ydot = [None]*self.size        
        ydot[0] = -(self.gNa*m**3*h*(V-self.vNa)+self.gK*n**4*(V-self.vK)+self.gL*(V-self.vL))        

        return ydot
    
    def u_exp_coeffs(self, uh):
        V = uh.values.sub(0)
        
        am = -0.1*(V+50.)/(ufl.exp(-(V+50.)/10.)-1.)
        an = -0.01*(V+65.)/(ufl.exp(-(V+65.)/10.)-1.)
        ah = 0.07*ufl.exp(-(V+75)/20.)
        bm = 4.0*ufl.exp(-(V+75)/18.)
        bn = 0.125*ufl.exp(-(V+75)/80.)
        bh = 1.0/(ufl.exp(-(V+45.)/10.)+1.)
        
        yinf = [None]*self.size
        tau = [None]*self.size
        
        tau[1] = 1/(am+bm)
        yinf[1] = am/(am+bm)
        tau[2] = 1/(an+bn)
        yinf[2] = an/(an+bn)
        tau[3] = 1/(ah+bh)
        yinf[3] = ah/(ah+bh)

        return yinf,tau
    
    def initial_values(self):
        return [self.V0,self.m0,self.n0,self.h0]
    

class RogersMcCulloch(IonicModel):
    def __init__(self):
        super(RogersMcCulloch,self).__init__()

        self.size = 2

        self.G = 1.5 # mS/cm^2
        self.v_th = 13. # mV
        self.v_p = 100. # mV
        self.eta_1 = 4.4 # mS/cm^2
        self.eta_2 = 0.012 # unitless
        self.eta_3 = 1. # unitless

        self.V0 = 0.
        self.w0 = 0.

        self.f_stiff_args = []
    
    def f(self,uh):
        V, w = uh.values.split()

        ydot = [None]*self.size

        ydot[0] = -(self.G*V*(1.-V/self.v_th)*(1.-V/self.v_p)+self.eta_1*V*w)
        ydot[1] = self.eta_2*(V/self.v_p-self.eta_3*w)
        
        return ydot
    
    def f_nonstiff(self,uh):
        return self.f(uh)
    
    def f_stiff(self, uh):
        ydot = [None]*self.size
        return ydot
    
    def f_expl(self,uh):
        return self.f(uh)
    
    def u_exp_coeffs(self, uh):
        yinf = [None]*self.size
        tau = [None]*self.size
        
        return yinf,tau    
    
    def initial_values(self):
        return [self.V0,self.w0]
    