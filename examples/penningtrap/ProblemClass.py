import numpy as np

from pySDC.Problem import ptype
from pySDC.datatype_classes.particles import particles, fields, acceleration

class penningtrap_single(ptype):
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
        assert 'omega_B' in cparams # B field frequency
        assert 'omega_E' in cparams # E field frequency
        assert 'alpha' in cparams   # mass to charge ratio
        assert 'eps' in cparams     # +/-1
        assert 'u0' in cparams      # initial position and velocity

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # set nparts to one (lonely particle, you know)
        self.nparts = 1
        # invoke super init, passing nparts, dtype_u and dtype_f
        super(penningtrap_single,self).__init__(self.nparts, dtype_u, dtype_f)


    def eval_f(self,part,t):
        """
        Routine to compute the E field (named f for consistency with the original PEPC version)

        Args:
            t: current time (not used here)
            part: the current particle
        Returns:
            E field for the particle (external only)
        """

        Emat = np.diag([1,1,-2])
        f = fields(self.nparts)
        f.elec.values[:] = -self.eps*self.omega_E**2/self.alpha*np.dot(Emat,part.pos.values)
        f.magn.values[:] = self.omega_B/self.alpha*np.array([0,0,1])

        return f


    def u_exact(self,t):
        """
        Routine to compute the exact trajectory at time t

        Args:
            t: current time
        Returns:
            particle type containing the exact position and velocity
        """

        # some abbreviations
        wE = self.omega_E
        wB = self.omega_B
        eps = self.eps
        N = self.nparts
        u0 = self.u0
        u = particles(N)

        wbar = np.sqrt(-2*eps)*wE

        # position and velocity in z direction is easy to compute
        u.pos.values[2] = u0[0,2]*np.cos(wbar*t) + u0[1,2]/wbar*np.sin(wbar*t)
        u.vel.values[2] = -u0[0,2]*wbar*np.sin(wbar*t) + u0[1,2]*np.cos(wbar*t)

        # define temp. variables to compute complex position
        Op = 1/2*(wB + np.sqrt(wB**2+4*eps*wE**2))
        Om = 1/2*(wB - np.sqrt(wB**2+4*eps*wE**2))
        Rm = (Op*u0[0,0]+u0[1,1])/(Op-Om)
        Rp = u0[0,0] - Rm
        Im = (Op*u0[0,1]-u0[1,0])/(Op-Om)
        Ip = u0[0,1] - Im

        # compute position in complex notation
        w = np.complex(Rp,Ip)*np.exp(-np.complex(0,Op*t)) + np.complex(Rm,Im)*np.exp(-np.complex(0,Om*t))
        # compute velocity as time derivative of the position
        dw = -1j*Op*np.complex(Rp,Ip)*np.exp(-np.complex(0,Op*t)) - 1j*Om*np.complex(Rm,Im)*np.exp(-np.complex(0,Om*t))

        # get the appropriate real and imaginary parts
        u.pos.values[0] = w.real
        u.vel.values[0] = dw.real
        u.pos.values[1] = w.imag
        u.vel.values[1] = dw.imag

        return u


    def build_f(self,f,part,t):
        """
        Helper function to assemble the correct right-hand side out of B and E field

        Args:
            f: the field values
            part: the current particle data
            t: the current time
        Returns:
            correct RHS of type acceleration
        """

        assert type(part) == particles

        rhs = acceleration(self.nparts)
        rhs.values[:] = self.alpha*(f.elec.values + np.cross(part.vel.values,f.magn.values))

        return rhs


    def boris_solver(self,c,dt,old_fields,new_fields,oldvel):
        """
        The actual Boris solver for static B fields, extended by the c-term

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
        a = self.alpha
        Emean = 1/2*(old_fields.elec + new_fields.elec)
        # pre-velocity, separated by the electric forces (and the c term)
        vm = oldvel.values + dt/2*a*Emean.values + c.values/2
        # rotation
        t = dt/2*a*new_fields.magn.values
        s = 2*t/(1+np.linalg.norm(t,2)**2)
        vp = vm + np.cross(vm+np.cross(vm,t),s)
        # post-velocity
        vel = particles.velocity(self.nparts)
        vel.values[:] = vp + dt/2*a*Emean.values + c.values/2

        return vel

    def dump_timestep(self,u,f):
        """
        Dummy dumping routine
        """
        pass