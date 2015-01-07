import numpy as np

from pySDC.Problem import ptype
from pySDC.datatype_classes.particles import particles, fields, acceleration

class planewave_single(ptype):
    """
    Example implementing a single particle in a penning trap

    Attributes:
        nparts: number of particles (needs to be 1 here)
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'delta' in cparams   # polarization
        assert 'a0' in cparams      # normalized amplitude
        assert 'u0' in cparams      # initial position and velocity

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # set nparts to one (lonely particle, you know)
        self.nparts = 1
        # invoke super init, passing nparts, dtype_u and dtype_f
        super(planewave_single,self).__init__(self.nparts, dtype_u, dtype_f)


    def eval_f(self,part,t):
        """
        Routine to compute the electric and magnetic fields

        Args:
            t: current time
            part: the current particle
        Returns:
            E and B field for the particle (external only)
        """

        f = fields(self.nparts)

        # f.elec.values[0] = 0
        # f.elec.values[1] = self.delta*self.a0*np.sin(t-part.pos.values[0])
        # f.elec.values[2] = -(1-self.delta**2)**(1/2)*self.a0*np.cos(t-part.pos.values[0])
        #
        # f.magn.values[0] = 0
        # f.magn.values[1] = (1-self.delta**2)**(1/2)*self.a0*np.cos(t-part.pos.values[0])
        # f.magn.values[2] = self.delta*self.a0*np.sin(t-part.pos.values[0])

        R = np.linalg.norm(part.pos.values,2)
        f.elec.values[0] = 0.02/(R**3)*part.pos.values[0]
        f.elec.values[1] = 0.02/(R**3)*part.pos.values[1]
        f.elec.values[2] = 0

        f.magn.values[0] = 0
        f.magn.values[1] = 0
        f.magn.values[2] = R


        # print(f.elec.values,f.magn.values)

        return f


    def u_exact(self,t):
        """
        Routine to compute the exact trajectory at time t

        Args:
            t: current time
        Returns:
            particle type containing the exact position and velocity
        """

        u0 = self.u0
        # some abbreviations
        u = particles(1)

        # # we need a Newton iteration to get x, the rest will follow...
        # x1 = 0
        #
        # Fx = x1 - 1/4*self.a0**2*(t-x1 + 1/2*(2*self.delta**2-1)*np.sin(2*t-2*x1))
        # dFx = 1 + 1/4*self.a0**2*(1 + (2*self.delta**2-1)*np.cos(2*t-2*x1))
        #
        # res = abs(Fx)
        # while res > 1E-12:
        #     print(res)
        #     x1 -= Fx/dFx
        #     Fx = x1 - 1/4*self.a0**2*(t-x1 + 1/2*(2*self.delta**2-1)*np.sin(2*t-2*x1))
        #     dFx = 1 + 1/4*self.a0**2*(1 + (2*self.delta**2-1)*np.cos(2*t-2*x1))
        #     res = abs(Fx)
        #
        # Phi = t-x1
        #
        # u.pos.values[0] = 1/4*self.a0**2*(Phi + 1/2*(2*self.delta**2-1)*np.sin(2*Phi))
        # u.pos.values[1] = self.delta*self.a0*np.sin(Phi)
        # u.pos.values[2] = -(1-self.delta**2)**(1/2)*self.a0*np.cos(Phi)
        #
        # u.vel.values[0] = 0
        # u.vel.values[1] = 0
        # u.vel.values[2] = 0

        u.pos.values[0] = u0[0][0]
        u.pos.values[1] = u0[0][1]
        u.pos.values[2] = u0[0][2]

        u.vel.values[0] = u0[1][0]
        u.vel.values[1] = u0[1][1]
        u.vel.values[2] = u0[1][2]

        u.q[:] = u0[2][0]
        u.m[:] = u0[3][0]

        return u


    def build_f(self,f,part,t):
        """
        Helper function to assemble the correct right-hand side out of B and E field

        Args:
            f: wannabe right-hand side, actually the E field
            part: particle data
            t: current time
        Returns:
            correct RHS of type acceleration
        """

        assert type(part) == particles
        rhs = acceleration(self.nparts)
        rhs.values[:] = part.q[:]/part.m[:]*(f.elec.values + np.cross(part.vel.values,f.magn.values))

        return rhs


    def boris_solver(self,c,dt,old_fields,new_fields,oldvel):
        """
        The actual Boris solver for non-static B fields, extended by the c-term (now modified)

        Args:
            c: the c term gathering the known values from the previous iteration
            dt: the (probably scaled) time step size
            old_fields: the field values at the previous node m
            new_fields: the field values at the current node m+1
            oldvel: the velocity at the previous node m
        Returns:
            the velocity at the (m+1)th node
        """

        assert type(oldvel) == particles.velocity
        a = 1 # fixme: this boris assumes charge to mass ratio of 1
        c.values += dt*a*1/2*np.cross(oldvel.values,old_fields.magn.values-new_fields.magn.values)
        Emean = 1/2*(old_fields.elec + new_fields.elec)
        # Bmean = 1/2*(old_fields.magn + new_fields.magn)
        # Emean = old_fields.elec
        Bmean = new_fields.magn
        # pre-velocity, separated by the electric forces (and the c term)
        vm = oldvel.values + dt/2*a*Emean.values + c.values/2
        # rotation
        t = dt/2*a*Bmean.values
        s = 2*t/(1+np.linalg.norm(t,2)**2)
        vp = vm + np.cross(vm+np.cross(vm,t),s)
        # post-velocity
        vel = particles.velocity(self.nparts)
        vel.values[:] = vp + dt/2*a*Emean.values + c.values/2

        return vel