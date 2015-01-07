import numpy as np

from pySDC.Problem import ptype
from pySDC.datatype_classes.particles import particles, fields, acceleration

class penningtrap(ptype):
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
        assert 'u0' in cparams      # initial position and velocity
        assert 'nparts' in cparams  # number of particles
        assert 'sig' in cparams     # smoothing parameter for Coulomb interaction

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # set number of particles
        self.nparts = cparams['nparts']
        # invoke super init, passing nparts, dtype_u and dtype_f
        super(penningtrap,self).__init__(self.nparts, dtype_u, dtype_f)


    def __get_interactions(self,part):
        """
        Routine to compute the particle-particle interaction, assuming q = 1 for all particles

        Args:
            part: the particles
        Returns:
            the internal E field for each particle

        """

        N = self.nparts

        Efield = np.zeros(3*N)

        for i in range(N):
            for j in range(N):
                dist2 = np.linalg.norm(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3],2)**2+self.sig**2
                contrib = part.q[j]*(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3]) / dist2**(3/2)

                Efield[3*i  ] += contrib[0]
                Efield[3*i+1] += contrib[1]
                Efield[3*i+2] += contrib[2]

        return Efield


    def eval_f(self,part,t):
        """
        Routine to compute the E and B fields (named f for consistency with the original PEPC version)

        Args:
            t: current time (not used here)
            part: the particles
        Returns:
            Fields for the particles (internal and external)
        """

        N = self.nparts

        Emat = np.diag([1,1,-2])
        f = fields(self.nparts)

        f.elec.values = self.__get_interactions(part)


        for n in range(N):
            f.elec.values[3*n:3*n+3] += self.omega_E**2 / (part.q[n]/part.m[n]) * np.dot(Emat,part.pos.values[
                                                                                              3*n:3*n+3])
            f.magn.values[3*n:3*n+3] = self.omega_B * np.array([0,0,1])

        return f


    def u_init(self):
        """
        Routine to compute the starting values for the particles

        Returns:
            particle set filled with initial data
        """

        u0 = self.u0
        N = self.nparts

        u = particles(N)

        if u0[2][0] is not 1 or u0[3][0] is not 1:
            print('Error: so far only q = m = 1 is implemented (I think)')
            exit()

        # set first particle to u0
        u.pos.values[0] = u0[0][0]
        u.pos.values[1] = u0[0][1]
        u.pos.values[2] = u0[0][2]
        u.vel.values[0] = u0[1][0]
        u.vel.values[1] = u0[1][1]
        u.vel.values[2] = u0[1][2]

        u.q[0] = u0[2][0]
        u.m[0] = u0[3][0]

        # initialize random seed
        np.random.seed(N)

        comx = u.pos.values[0]
        comy = u.pos.values[1]
        comz = u.pos.values[2]

        for n in range(1,N):

            # draw 3 random variables in [-0.1,0.1] to shift positions
            r = 0.002*np.random.random_sample(3)-0.001
            u.pos.values[3*n  ] = r[0]+u0[0][0]
            u.pos.values[3*n+1] = r[1]+u0[0][1]
            u.pos.values[3*n+2] = r[2]+u0[0][2]

            # draw 3 random variables in [-5,5] to shift velocities
            r = 10*np.random.random_sample(3)-5
            u.vel.values[3*n  ] = r[0]+u0[1][0]
            u.vel.values[3*n+1] = r[1]+u0[1][1]
            u.vel.values[3*n+2] = r[2]+u0[1][2]

            u.q[n] = u0[2][0]
            u.m[n] = u0[3][0]

            # gather positions to check center
            comx += u.pos.values[3*n  ]
            comy += u.pos.values[3*n+1]
            comz += u.pos.values[3*n+2]

        print('Center of positions:',comx/N,comy/N,comz/N)

        return u



    def u_exact(self,t):
        """
        Routine to compute the exact trajectory at time t (only for single-particle setup)

        Args:
            t: current time
        Returns:
            particle type containing the exact position and velocity
        """

        # some abbreviations
        wE = self.omega_E
        wB = self.omega_B
        N = self.nparts
        u0 = self.u0

        assert N == 1

        u = particles(1)

        wbar = np.sqrt(2)*wE

        # position and velocity in z direction is easy to compute
        u.pos.values[2] = u0[0][2]*np.cos(wbar*t) + u0[1][2]/wbar*np.sin(wbar*t)
        u.vel.values[2] = -u0[0][2]*wbar*np.sin(wbar*t) + u0[1][2]*np.cos(wbar*t)

        # define temp. variables to compute complex position
        Op = 1/2*(wB + np.sqrt(wB**2-4*wE**2))
        Om = 1/2*(wB - np.sqrt(wB**2-4*wE**2))
        Rm = (Op*u0[0][0]+u0[1][1])/(Op-Om)
        Rp = u0[0][0] - Rm
        Im = (Op*u0[0][1]-u0[1][0])/(Op-Om)
        Ip = u0[0][1] - Im

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
            part: the current particles data
            t: the current time
        Returns:
            correct RHS of type acceleration
        """

        assert type(part) == particles

        N = self.nparts

        rhs = acceleration(self.nparts)

        for n in range(N):
            rhs.values[3*n:3*n+3] = part.q[n]/part.m[n]*(f.elec.values[3*n:3*n+3] + np.cross(part.vel.values[3*n:3*n+3],
                                                                                    f.magn.values[3*n:3*n+3]))

        return rhs


    def boris_solver(self,c,dt,old_fields,new_fields,old_parts):
        """
        The actual Boris solver for static (!) B fields, extended by the c-term

        Args:
            c: the c term gathering the known values from the previous iteration
            dt: the (probably scaled) time step size
            old_fields: the field values at the previous node m
            new_fields: the field values at the current node m+1
            old_parts: the particles at the previous node m
        Returns:
            the velocities at the (m+1)th node
        """

        N = self.nparts
        vel = particles.velocity(N)

        Emean = 1/2*(old_fields.elec + new_fields.elec)

        for n in range(N):

            a = old_parts.q[n]/old_parts.m[n]

            c.values[3*n:3*n+3] += dt/2*a*np.cross(old_parts.vel.values[3*n:3*n+3],
                                                   old_fields.magn.values[3*n:3*n+3]-new_fields.magn.values[3*n:3*n+3])

            # pre-velocity, separated by the electric forces (and the c term)
            vm = old_parts.vel.values[3*n:3*n+3] + dt/2*a*Emean.values[3*n:3*n+3] + c.values[3*n:3*n+3]/2
            # rotation
            t = dt/2* a * new_fields.magn.values[3*n:3*n+3]
            s = 2*t/(1+np.linalg.norm(t,2)**2)
            vp = vm + np.cross(vm+np.cross(vm,t),s)
            # post-velocity
            vel.values[3*n:3*n+3] = vp + dt/2*a* Emean.values[3*n:3*n+3] + c.values[3*n:3*n+3]/2

        return vel