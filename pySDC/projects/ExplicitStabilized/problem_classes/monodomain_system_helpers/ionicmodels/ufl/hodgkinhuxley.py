import ufl
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import IonicModel
from pySDC.projects.ExplicitStabilized.problem_classes.monodomain_system_helpers.ionicmodels.ufl.ionicmodel import NV_Ith_S

# Non stiff with rho_max ~ 40
class HodgkinHuxley(IonicModel):
    def __init__(self,scale):
        super(HodgkinHuxley,self).__init__(scale)

        self.size = 4        

        # Set values of constants
        self.AC_g_L = 0.3
        self.AC_Cm = 1.0
        self.AC_E_R = -75.0
        self.AC_E_K = self.AC_E_R - 12.0
        self.AC_g_K = 36.0
        self.AC_E_Na = self.AC_E_R + 115.0
        self.AC_g_Na = 120.0
        self.AC_E_L = self.AC_E_R + 10.613

        # list of indeces needed to compute or which are affected by the given function
        self.f_nonstiff_args = [0,1,2,3]
        self.f_stiff_args = [0,1]
        self.f_expl_args = [0,1,2,3]
        self.f_exp_args = [0,1,2,3]

        # list of indeces affected by a given function
        self.f_stiff_indeces = [1]
        self.f_nonstiff_indeces = [0,2,3]
        self.f_exp_indeces = [1,2,3]
        self.f_expl_indeces = [0]    

    def initial_values(self):

        y0 = [None]*self.size
        y0[0] = -75.0
        y0[1] = 0.05
        y0[2] = 0.595
        y0[3] = 0.317

        return y0
    
    @property
    def f(self):

        y = self.y
        ydot = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -self.scale*( AV_i_Na + AV_i_K + AV_i_L)

        return self.expression_list(ydot)
    
    @property
    def f_expl(self):
        y = self.y
        ydot = [None]*self.size            
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -self.scale*( AV_i_Na + AV_i_K + AV_i_L)

        return self.expression_list(ydot)

    @property
    def f_exp(self):

        y = self.y
        ydot = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)

        return self.expression_list(ydot)
    
    @property
    def phi_f_exp(self):

        y = self.y
        lmbda = [None]*self.size
        yinf = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        lmbda[3] = -(AV_alpha_n+AV_beta_n)
        yinf[3] = -AV_alpha_n/lmbda[3]
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        lmbda[2] = -(AV_alpha_h+AV_beta_h)
        yinf[2] = -AV_alpha_h/lmbda[2]
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        lmbda[1] = -(AV_alpha_m+AV_beta_m)
        yinf[1] = -AV_alpha_m/lmbda[1]
        
        return self.expression_list(self.apply_phi(lmbda,yinf))    
    
    @property
    def lmbda_yinf_exp(self):

        y = self.y
        lmbda = [None]*self.size
        yinf = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        lmbda[3] = -(AV_alpha_n+AV_beta_n)
        yinf[3] = -AV_alpha_n/lmbda[3]
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        lmbda[2] = -(AV_alpha_h+AV_beta_h)
        yinf[2] = -AV_alpha_h/lmbda[2]
        
        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        lmbda[1] = -(AV_alpha_m+AV_beta_m)
        yinf[1] = -AV_alpha_m/lmbda[1]
        
        return self.expression_list(lmbda), self.expression_list(yinf)    
    
    @property
    def lmbda_exp(self):
        lmbda_expr, yinf_expr = self.lmbda_yinf_exp()
        return lmbda_expr
    
    
    @property
    def f_nonstiff(self):
        y = self.y
        ydot = [None]*self.size
            
        AV_alpha_n = (-0.01) * (NV_Ith_S(y, 0) + 65.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 65.0)) / 10.0) - 1.0)
        AV_beta_n = 0.125 * ufl.exp((NV_Ith_S(y, 0) + 75.0) / 80.0)
        ydot[3] = AV_alpha_n * (1.0 - NV_Ith_S(y, 3)) - AV_beta_n * NV_Ith_S(y, 3)
        
        AV_alpha_h = 0.07 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 20.0)
        AV_beta_h = 1.0 / (ufl.exp((-(NV_Ith_S(y, 0) + 45.0)) / 10.0) + 1.0)
        ydot[2] = AV_alpha_h * (1.0 - NV_Ith_S(y, 2)) - AV_beta_h * NV_Ith_S(y, 2)             
                
        AV_i_K = self.AC_g_K * pow(NV_Ith_S(y, 3), 4.0) * (NV_Ith_S(y, 0) - self.AC_E_K)        
        AV_i_Na = self.AC_g_Na * pow(NV_Ith_S(y, 1), 3.0) * NV_Ith_S(y, 2) * (NV_Ith_S(y, 0) - self.AC_E_Na)        
        AV_i_L = self.AC_g_L * (NV_Ith_S(y, 0) - self.AC_E_L)
        ydot[0] = -self.scale*( AV_i_Na + AV_i_K + AV_i_L)

        return self.expression_list(ydot)
    
    @property
    def f_stiff(self):
        y = self.y
        ydot = [None]*self.size              

        AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
        AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
        ydot[1] = AV_alpha_m * (1.0 - NV_Ith_S(y, 1)) - AV_beta_m * NV_Ith_S(y, 1)            

        return self.expression_list(ydot)

    # @property
    # def phi_f_stiff(self):

    #     y = self.y
    #     lmbda = [None]*self.size            
    #     yinf = [None]*self.size            
        
    #     AV_alpha_m = (-0.1) * (NV_Ith_S(y, 0) + 50.0) / (ufl.exp((-(NV_Ith_S(y, 0) + 50.0)) / 10.0) - 1.0)
    #     AV_beta_m = 4.0 * ufl.exp((-(NV_Ith_S(y, 0) + 75.0)) / 18.0)
    #     lmbda[1] = -(AV_alpha_m+AV_beta_m)
    #     yinf[1] = -AV_alpha_m/lmbda[1]    

    #     return self.expression_list(self.apply_phi(lmbda,yinf))        
