from dolfinx import fem
import ufl

def NV_Ith_S(y,k):
    return y.sub(k)
    
class IonicModel:
    def __init__(self,scale):
        self.scale = scale
    
    def set_domain(self,domain):
        self.domain = domain

    def set_y(self,y):
        self.y = y

    def set_dt(self,dt):
        self.dt = dt

    def set_V(self,V):
        self.V = V

    def f_from_lmbda_yinf(self,lmbda,yinf):
        ydot = [None]*self.size
        for i in range(self.size):
            if lmbda[i] is not None:
                ydot[i] = lmbda[i]*(NV_Ith_S(self.y,i)-yinf[i])
        return ydot
    
    def apply_phi(self,lmbda,yinf):
        phi_f_exp = [None]*self.size
        for i in range(self.size):
            if lmbda[i] is not None:
                phi_f_exp[i] = ((ufl.exp(self.dt*lmbda[i])-1.)/(self.dt))*(NV_Ith_S(self.y, i)-yinf[i])
        return phi_f_exp

    def expression_list(self,y):
        y_expr = [None]*self.size
        for i in range(self.size):
            if y[i] is not None:
                y_expr[i] = fem.Expression(y[i],self.V.element.interpolation_points())   
        return y_expr