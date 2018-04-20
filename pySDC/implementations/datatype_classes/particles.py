
import numpy as np
import copy as cp

from pySDC.core.Errors import DataError


class particles(object):
    """
    Particle data type for particles in any dimensions

    Attributes:
        pos: contains the positions of all particles
        vel: contains the velocities of all particles
    """

    class position(object):
        """
        Position data type for particles in any dimensions

        Attributes:
            values (np.ndarray): array with position values
        """

        def __init__(self, init=None, val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another position object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another position, do a deepcopy (init by copy)
            if isinstance(init, type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create position object with val as initial value
            elif isinstance(init, int) or isinstance(init, tuple):
                self.values = np.empty(init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))

        def __add__(self, other):
            """
            Overloading the addition operator for position types

            Args:
                other (position): position object to be added
            Raises:
                DataError: if other is not a position object
            Returns:
                position: sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a + b changes a as well!
                pos = particles.position(self.values.shape)
                pos.values = self.values + other.values
                return pos
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

        def __sub__(self, other):
            """
            Overloading the subtraction operator for position types

            Args:
                other (position): position object to be subtracted
            Raises:
                DataError: if other is not a position object
            Returns:
                position: differences between caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a - b changes a as well!
                pos = particles.position(self.values.shape)
                pos.values = self.values - other.values
                return pos
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for position types

            Args:
                other (float): factor
            Raises:
                DataError: is other is not a float
            Returns:
                position: original values scaled by factor
            """

            if isinstance(other, float):
                # create new position
                pos = particles.position(self.values.shape)
                pos.values = self.values * other
                return pos
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

        def __abs__(self):
            """
            Overloading the abs operator for position types

            Returns:
                float: absolute maximum of all position values
            """
            return np.amax(np.absolute(self.values))

    class velocity(object):
        """
        Velocity data type for particles in any dimensions

        Attributes:
            values (np.ndarray): array with velocity values
        """

        def __init__(self, init=None, val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another velocity object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another velocity, do a deepcopy (init by copy)
            if isinstance(init, type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create velocity object with val as initial value
            elif isinstance(init, int) or isinstance(init, tuple):
                self.values = np.empty(init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))

        def __add__(self, other):
            """
            Overloading the addition operator for velocity types

            Args:
                other: velocity object to be added
            Raises:
                DataError: if other is not a velocity object
            Returns:
                velocity: sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a + b changes a as well!
                vel = particles.velocity(self.values.shape)
                vel.values = self.values + other.values
                return vel
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

        def __sub__(self, other):
            """
            Overloading the subtraction operator for velocity types

            Args:
                other: velocity object to be subtracted
            Raises:
                DataError: if other is not a velocity object
            Returns:
                velocity: differences between caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a - b changes a as well!
                vel = particles.velocity(self.values.shape)
                vel.values = self.values - other.values
                return vel
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for velocity types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                position: original values scaled by factor, transformed to position
            """

            if isinstance(other, float):
                # create new position, interpret float factor as time (time x velocity = position)
                pos = particles.position(self.values.shape)
                pos.values = self.values * other
                return pos
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

        def __abs__(self):
            """
            Overloading the abs operator for velocity types

            Returns:
                float: absolute maximum of all velocity values
            """
            # FIXME: is this a good idea for multiple particles?
            return np.amax(np.absolute(self.values))

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another particle object
            val: initial tuple of values for position and velocity (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another particles object, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.pos = particles.position(init.pos)
            self.vel = particles.velocity(init.vel)
            self.q = cp.deepcopy(init.q)
            self.m = cp.deepcopy(init.m)
        # if init is a number, create particles object and pick the corresponding initial values
        elif isinstance(init, int):
            if isinstance(val, int) or val is None:
                self.pos = particles.position(init, val=val)
                self.vel = particles.velocity(init, val=val)
                self.q = np.zeros(init)
                self.q[:] = val
                self.m = np.zeros(init)
                self.m[:] = val
            elif isinstance(val, tuple) and len(val) == 4:
                self.pos = particles.position(init, val=val[0])
                self.vel = particles.velocity(init, val=val[1])
                self.q = np.zeros(init)
                self.q[:] = val[2]
                self.m = np.zeros(init)
                self.m[:] = val[3]
            else:
                raise DataError('type of val is wrong, got %s', val)
        elif isinstance(init, tuple):
            if isinstance(val, int) or val is None:
                self.pos = particles.position(init, val=val)
                self.vel = particles.velocity(init, val=val)
                self.q = np.zeros(init[-1])
                self.q[:] = val
                self.m = np.zeros(init[-1])
                self.m[:] = val
            elif isinstance(val, tuple) and len(val) == 4:
                self.pos = particles.position(init, val=val[0])
                self.vel = particles.velocity(init, val=val[1])
                self.q = np.zeros(init[-1])
                self.q[:] = val[2]
                self.m = np.zeros(init[-1])
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
            p = particles(self.pos.values.shape)
            p.pos = self.pos + other.pos
            p.vel = self.vel + other.vel
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
            p = particles(self.pos.values.shape)
            p.pos = self.pos - other.pos
            p.vel = self.vel - other.vel
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
            p = particles(self.pos.values.shape)
            p.pos = other * self.pos
            p.vel.values = other * self.vel.values
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


class acceleration(object):
    """
    Acceleration data type for particles in any dimensions

    Attributes:
        values (np.ndarray): array with acceleration values
    """

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another acceleration object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another particles object, do a deepcopy (init by copy)
        if isinstance(init, acceleration):
            self.values = cp.deepcopy(init.values)
        # if init is a number, create acceleration object with val as initial value
        elif isinstance(init, int) or isinstance(init, tuple):
            self.values = np.empty(init)
            self.values[:] = val
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for acceleration types

        Args:
            other (acceleration): acceleration object to be added
        Raises:
            DataError: if other is not a acceleration object
        Returns:
            acceleration: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new acceleration, since otherwise c = a + b changes a as well!
            acc = acceleration(self.values.shape)
            acc.values = self.values + other.values
            return acc
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for acceleration types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            velocity: original values scaled by factor, tranformed to velocity
        """

        if isinstance(other, float):
            # create new velocity, interpret float factor as time (time x acceleration = velocity)
            vel = particles.velocity(self.values.shape)
            vel.values = self.values * other
            return vel
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))


class imex_acceleration(object):
    """
    IMEX Acceleration data type for particles in any dimensions

    This acceleration data type can hold an implicit and an explicit part

    Attributes:
        impl (acceleration): acceleration values for the implicit part
        expl (acceleration): acceleration values for the explicit part
    """
    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another imex_acceleration object object
            val: initial tuple of values for both parts of the acceleration
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another imex_acceleration object, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = acceleration(init.impl)
            self.expl = acceleration(init.expl)
        # if init is a number, create imex_acceleration object and pick the corresponding initial values
        elif isinstance(init, int) or isinstance(init, tuple):
            if isinstance(val, int) or val is None:
                self.impl = acceleration(init, val=val)
                self.expl = acceleration(init, val=val)
            elif isinstance(val, tuple) and len(val) == 2:
                self.impl = acceleration(init, val=val[0])
                self.expl = acceleration(init, val=val[1])
            else:
                raise DataError('wrong type of val, got %s' % val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


class fields(object):
    """
    Field data type for any dimensions

    Attributes:
        elec: contains the electric field
        magn: contains the magnetic field
    """

    class electric(object):
        """
        Electric field data type in any dimensions

        Attributes:
            values (np.ndarray): array with electric values
        """

        def __init__(self, init=None, val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another electric object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another electric object, do a deepcopy (init by copy)
            if isinstance(init, type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create electric object with val as initial value
            elif isinstance(init, int) or isinstance(init, tuple):
                self.values = np.empty(init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))

        def __add__(self, other):
            """
            Overloading the addition operator for electric types

            Args:
                other (electric): electric object to be added
            Raises:
                DataError: if other is not a electric object
            Returns:
                electric: sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new electric, since otherwise c = a + b changes a as well!
                E = fields.electric(self.values.shape)
                E.values = self.values + other.values
                return E
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

        def __sub__(self, other):
            """
            Overloading the subtraction operator for electric types

            Args:
                other (electric): electric object to be subtracted
            Raises:
                DataError: if other is not a electric object
            Returns:
                electric: difference of caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new electric, since otherwise c = a + b changes a as well!
                E = fields.electric(self.values.shape)
                E.values = self.values - other.values
                return E
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for electric types

            Args:
                other (float): factor
            Raises:
                DataError: is other is not a float
            Returns:
                electric: original values scaled by factor
            """

            if isinstance(other, float):
                # create new electric, no specific interpretation of float factor
                E = fields.electric(self.values.shape)
                E.values = self.values * other
                return E
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    class magnetic(object):
        """
        Magnetic field data type in any dimensions

        Attributes:
            values (np.ndarray): array with magnetic values
        """

        def __init__(self, init=None, val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another magnetic object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another magnetic object, do a deepcopy (init by copy)
            if isinstance(init, type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create magnetic object with val as initial value
            elif isinstance(init, int) or isinstance(init, tuple):
                self.values = np.empty(init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))

        def __add__(self, other):
            """
            Overloading the addition operator for magnetic types

            Args:
                other (magnetic): magnetic object to be added
            Raises:
                DataError: if other is not a magnetic object
            Returns:
                magnetic: sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new magnetic, since otherwise c = a + b changes a as well!
                M = fields.magnetic(self.values.shape)
                M.values = self.values + other.values
                return M
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

        def __sub__(self, other):
            """
            Overloading the subrtaction operator for magnetic types

            Args:
                other (magnetic): magnetic object to be subtracted
            Raises:
                DataError: if other is not a magnetic object
            Returns:
                magnetic: difference of caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new magnetic, since otherwise c = a + b changes a as well!
                M = fields.magnetic(self.values.shape)
                M.values = self.values - other.values
                return M
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for magnetic types

            Args:
                other (float): factor
            Raises:
                DataError: is other is not a float
            Returns:
                electric: original values scaled by factor, transformed to electric
            """

            if isinstance(other, float):
                # create new magnetic, no specific interpretation of float factor
                M = fields.magnetic(self.values.shape)
                M.values = self.values * other
                return M
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another fields object
            val: initial tuple of values for electric and magnetic (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another fields object, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.elec = fields.electric(init.elec)
            self.magn = fields.magnetic(init.magn)
        # if init is a number, create fields object and pick the corresponding initial values
        elif isinstance(init, int) or isinstance(init, tuple):
            if isinstance(val, int) or val is None:
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
            p = fields(self.elec.values.shape)
            p.elec = self.elec + other.elec
            p.magn = self.magn + other.magn
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
            p = fields(self.elec.values.shape)
            p.elec = self.elec - other.elec
            p.magn = self.magn - other.magn
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
            p = fields(self.elec.values.shape)
            p.elec = other * self.elec
            p.magn = other * self.magn
            return p
        else:
            raise DataError("Type error: cannot multiply %s with %s" % (type(other), type(self)))
