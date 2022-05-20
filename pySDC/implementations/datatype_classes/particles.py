import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class particles(object):
    """
    Particle data type for particles in 3 dimensions

    This data type can be used for particles in 3 dimensions with 3 position and 3 velocity values per particle

    Attributes:
        pos: contains the positions of all particles
        vel: contains the velocities of all particles
    """

    class position(mesh):
        pass

    class velocity(mesh):
        pass

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another particle object
            val: initial tuple of values for position and velocity (default: (None,None))
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another particles object, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.pos = particles.position(init.pos)
            self.vel = particles.velocity(init.vel)
            self.q = init.q.copy()
            self.m = init.m.copy()
        # if init is a number, create particles object and pick the corresponding initial values
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            if isinstance(val, int) or isinstance(val, float) or val is None:
                self.pos = particles.position(init, val=val)
                self.vel = particles.velocity(init, val=val)
                if isinstance(init[0], tuple):
                    self.q = np.zeros(init[0][-1])
                    self.m = np.zeros(init[0][-1])
                elif isinstance(init[0], int):
                    self.q = np.zeros(init[0])
                    self.m = np.zeros(init[0])
                self.q[:] = 1.0
                self.m[:] = 1.0
            elif isinstance(val, tuple) and len(val) == 4:
                self.pos = particles.position(init, val=val[0])
                self.vel = particles.velocity(init, val=val[1])
                if isinstance(init[0], tuple):
                    self.q = np.zeros(init[0][-1])
                    self.m = np.zeros(init[0][-1])
                elif isinstance(init[0], int):
                    self.q = np.zeros(init[0])
                    self.m = np.zeros(init[0])
                self.q[:] = val[2]
                self.m[:] = val[3]
            else:
                raise DataError('type of val is wrong, got %s', val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for particles types

        Args:
            other (particles): particles object to be added
        Raises:
            DataError: if other is not a particles object
        Returns:
            particles: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new particles, since otherwise c = a + b changes a as well!
            p = particles(self)
            p.pos[:] = self.pos + other.pos
            p.vel[:] = self.vel + other.vel
            p.m = self.m
            p.q = self.q
            return p
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for particles types

        Args:
            other (particles): particles object to be subtracted
        Raises:
            DataError: if other is not a particles object
        Returns:
            particles: differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new particles, since otherwise c = a - b changes a as well!
            p = particles(self)
            p.pos[:] = self.pos - other.pos
            p.vel[:] = self.vel - other.vel
            p.m = self.m
            p.q = self.q
            return p
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for particles types

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a particles object
        Returns:
            particles: scaled particle's velocity and position as new particle
        """

        if isinstance(other, float):
            # always create new particles
            p = particles(self)
            p.pos[:] = other * self.pos
            p.vel[:] = other * self.vel
            p.m = self.m
            p.q = self.q
            return p
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for particles types

        Returns:
            float: absolute maximum of abs(pos) and abs(vel) for all particles
        """
        abspos = abs(self.pos)
        absvel = abs(self.vel)
        return np.amax((abspos, absvel))

    def send(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """

        comm.send(self, dest=dest, tag=tag)
        return None

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.isend(self, dest=dest, tag=tag)

    def recv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        part = comm.recv(source=source, tag=tag)
        self.pos[:] = part.pos.copy()
        self.vel[:] = part.vel.copy()
        self.m = part.m.copy()
        self.q = part.q.copy()
        return None


class acceleration(mesh):
    pass


class fields(object):
    """
    Field data type for 3 dimensions

    This data type can be used for electric and magnetic fields in 3 dimensions

    Attributes:
        elec: contains the electric field
        magn: contains the magnetic field
    """

    class electric(mesh):
        pass

    class magnetic(mesh):
        pass

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another fields object
            val: initial tuple of values for electric and magnetic (default: (None,None))
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another fields object, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.elec = fields.electric(init.elec)
            self.magn = fields.magnetic(init.magn)
        # if init is a number, create fields object and pick the corresponding initial values
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            if isinstance(val, int) or isinstance(val, float) or val is None:
                self.elec = fields.electric(init, val=val)
                self.magn = fields.magnetic(init, val=val)
            elif isinstance(val, tuple) and len(val) == 2:
                self.elec = fields.electric(init, val=val[0])
                self.magn = fields.magnetic(init, val=val[1])
            else:
                raise DataError('wrong type of val, got %s' % val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for fields types

        Args:
            other (fields): fields object to be added
        Raises:
            DataError: if other is not a fields object
        Returns:
            fields: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new fields, since otherwise c = a - b changes a as well!
            p = fields(self)
            p.elec[:] = self.elec + other.elec
            p.magn[:] = self.magn + other.magn
            return p
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for fields types

        Args:
            other (fields): fields object to be subtracted
        Raises:
            DataError: if other is not a fields object
        Returns:
            fields: differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new fields, since otherwise c = a - b changes a as well!
            p = fields(self)
            p.elec[:] = self.elec - other.elec
            p.magn[:] = self.magn - other.magn
            return p
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the multiply with factor from right operator for fields types

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a fields object
        Returns:
            fields: scaled fields
        """

        if isinstance(other, float):
            # always create new fields, since otherwise c = a - b changes a as well!
            p = fields(self)
            p.elec[:] = other * self.elec
            p.magn[:] = other * self.magn
            return p
        else:
            raise DataError("Type error: cannot multiply %s with %s" % (type(other), type(self)))
