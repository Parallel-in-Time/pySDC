import numpy as np
import copy as cp

from pySDC.Errors import DataError


#FIXME: this may not work for multiple particles...
class particles():
    """
    Particle data type for particles in 3 dimensions

    This data type can be used for particles in 3 dimensions with 3 position and 3 velocity values per particle

    Attributes:
        pos: contains the positions of all particles
        vel: contains the velocities of all particles
    """

    class position():
        """
        Position data type for particles in 3 dimensions

        Attributes:
            values: array with 3 position values per particle (dim. 3*nparts)
        """

        def __init__(self,init=None,val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another position object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another position, do a deepcopy (init by copy)
            if isinstance(init,type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create position object with val as initial value
            elif isinstance(init,int):
                self.values = np.empty(3*init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))


        def __add__(self, other):
            """
            Overloading the addition operator for position types

            Args:
                other: position object to be added
            Raises:
                DataError: if other is not a position object
            Returns:
                sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a + b changes a as well!
                pos = particles.position(int(np.size(self.values)/3))
                pos.values = self.values + other.values
                return pos
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))


        def __sub__(self, other):
            """
            Overloading the subtraction operator for position types

            Args:
                other: position object to be subtracted
            Raises:
                DataError: if other is not a position object
            Returns:
                differences between caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a - b changes a as well!
                pos = particles.position(int(np.size(self.values)/3))
                pos.values = self.values - other.values
                return pos
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for position types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                position(!) object, original values scaled by factor
            """

            if isinstance(other, float):
                # create new position, interpret float factor as time (time x velocity = position)
                pos = particles.position(int(np.size(self.values)/3))
                pos.values = self.values*other
                return pos
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


        def __abs__(self):
            """
            Overloading the abs operator for position types

            Returns:
                absolute maximum of all position values
            """
            #FIXME: is this a good idea for multiple particles?
            return np.amax(np.absolute(self.values))


    class velocity():
        """
        Velocity data type for particles in 3 dimensions

        Attributes:
            values: array with 3 velocity values per particle (dim. 3*nparts)
        """

        def __init__(self,init=None,val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another velocity object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another velocity, do a deepcopy (init by copy)
            if isinstance(init,type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create velocity object with val as initial value
            elif isinstance(init,int):
                self.values = np.empty(3*init)
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
                sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a + b changes a as well!
                vel = particles.velocity(int(np.size(self.values)/3))
                vel.values = self.values + other.values
                return vel
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))


        def __sub__(self, other):
            """
            Overloading the subtraction operator for velocity types

            Args:
                other: velocity object to be subtracted
            Raises:
                DataError: if other is not a velocity object
            Returns:
                differences between caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new position, since otherwise c = a - b changes a as well!
                vel = particles.velocity(int(np.size(self.values)/3))
                vel.values = self.values - other.values
                return vel
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))


        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for velocity types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                position(!) object, original values scaled by factor
            """

            if isinstance(other, float):
                # create new position, interpret float factor as time (time x velocity = position)
                pos = particles.position(int(np.size(self.values)/3))
                pos.values = self.values*other
                return pos
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


        def __abs__(self):
            """
            Overloading the abs operator for velocity types

            Returns:
                absolute maximum of all velocity values
            """
            #FIXME: is this a good idea for multiple particles?
            return np.amax(np.absolute(self.values))


    def __init__(self,init=None,vals=(None,None,None,None)):
        """
        Initialization routine

        Args:
            init: can either be a number or another particle object
            vals: initial tuple of values for position and velocity (default: (None,None))
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another particles object, do a deepcopy (init by copy)
        if isinstance(init,type(self)):
            self.pos = particles.position(init.pos)
            self.vel = particles.velocity(init.vel)
            self.q = cp.deepcopy(init.q)
            self.m = cp.deepcopy(init.m)
        # if init is a number, create particles object and pick the corresponding initial values
        elif isinstance(init,int):
            self.pos = particles.position(init,val=vals[0])
            self.vel = particles.velocity(init,val=vals[1])
            self.q = np.zeros(int(np.size(self.pos.values)/3))
            self.q[:] = vals[2]
            self.m = np.zeros(int(np.size(self.pos.values)/3))
            self.m[:] = vals[3]
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


    def __add__(self, other):
        """
        Overloading the addition operator for particles types

        Args:
            other: particles object to be added
        Raises:
            DataError: if other is not a particles object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new particles, since otherwise c = a + b changes a as well!
            p = particles(int(np.size(self.pos.values)/3))
            p.pos = self.pos + other.pos
            p.vel = self.vel + other.vel
            p.m = self.m
            p.q = self.q
            return p
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))


    def __sub__(self, other):
        """
        Overloading the subtraction operator for particles types

        Args:
            other: particles object to be subtracted
        Raises:
            DataError: if other is not a particles object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new particles, since otherwise c = a - b changes a as well!
            p = particles(int(np.size(self.pos.values)/3))
            p.pos = self.pos - other.pos
            p.vel = self.vel - other.vel
            p.m = self.m
            p.q = self.q
            return p
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))

    def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for particle types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                particle
            """

            if isinstance(other, float):
                # create particle here!
                part = particles(int(np.size(self.pos.values)/3))
                part.vel.values = self.vel.values*other
                part.pos.values = self.pos.values*other
                part.m = self.m
                part.q = self.q
                return part
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


    def __abs__(self):
        """
        Overloading the abs operator for particles types

        Returns:
            absolute maximum of abs(pos) and abs(vel) for all particles
        """
        #FIXME: is this a good idea for multiple particles?
        abspos = abs(self.pos)
        absvel = abs(self.vel)
        return np.amax((abspos,absvel))


class acceleration():
    """
    Acceleration data type for particles in 3 dimensions

    Attributes:
        values: array with 3 acceleration values per particle (dim. 3*nparts)
    """


    def __init__(self,init=None,val=None):
        """
        Initialization routine

        Args:
            init: can either be a number or another acceleration object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another particles object, do a deepcopy (init by copy)
        if isinstance(init,acceleration):
            self.values = cp.deepcopy(init.values)
        # if init is a number, create acceleration object with val as initial value
        elif isinstance(init,int):
            self.values = np.empty(3*init)
            self.values[:] = val
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for acceleration types

        Args:
            other: acceleration object to be added
        Raises:
            DataError: if other is not a acceleration object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new acceleration, since otherwise c = a + b changes a as well!
            acc = acceleration(int(np.size(self.values)/3))
            acc.values = self.values + other.values
            return acc
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for acceleration types

        Args:
            other: float factor
        Raises:
            DataError: is other is not a float
        Returns:
            velocity(!) object, original values scaled by factor
        """

        if isinstance(other, float):
            # create new velocity, interpret float factor as time (time x acceleration = velocity)
            vel = particles.velocity(int(np.size(self.values)/3))
            vel.values = self.values*other
            return vel
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


class fields():
    """
    Field data type for 3 dimensions

    This data type can be used for electric and magnetic fields in 3 dimensions

    Attributes:
        elec: contains the electric field
        magn: contains the magnetic field
    """

    class electric():
        """
        Electric field data type in 3 dimensions

        Attributes:
            values: array with 3 field values per particle (dim. 3*nparts)
        """

        def __init__(self,init=None,val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another electric object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another electric object, do a deepcopy (init by copy)
            if isinstance(init,type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create electric object with val as initial value
            elif isinstance(init,int):
                self.values = np.empty(3*init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))

        def __add__(self, other):
            """
            Overloading the addition operator for electric types

            Args:
                other: electric object to be added
            Raises:
                DataError: if other is not a electric object
            Returns:
                sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new electric, since otherwise c = a + b changes a as well!
                E = fields.electric(int(np.size(self.values)/3))
                E.values = self.values + other.values
                return E
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))

        def __sub__(self, other):
            """
            Overloading the subtraction operator for electric types

            Args:
                other: electric object to be subtracted
            Raises:
                DataError: if other is not a electric object
            Returns:
                difference of caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new electric, since otherwise c = a + b changes a as well!
                E = fields.electric(int(np.size(self.values)/3))
                E.values = self.values - other.values
                return E
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))

        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for electric types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                electric object, original values scaled by factor
            """

            if isinstance(other, float):
                # create new electric, no specific interpretation of float factor
                E = fields.electric(int(np.size(self.values)/3))
                E.values = self.values*other
                return E
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


    class magnetic():
        """
        Magnetic field data type in 3 dimensions

        Attributes:
            values: array with 3 field values per particle (dim. 3*nparts)
        """

        def __init__(self,init=None,val=None):
            """
            Initialization routine

            Args:
                init: can either be a number or another magnetic object
                val: initial value (default: None)
            Raises:
                DataError: if init is none of the types above
            """

            # if init is another magnetic object, do a deepcopy (init by copy)
            if isinstance(init,type(self)):
                self.values = cp.deepcopy(init.values)
            # if init is a number, create magnetic object with val as initial value
            elif isinstance(init,int):
                self.values = np.empty(3*init)
                self.values[:] = val
            # something is wrong, if none of the ones above hit
            else:
                raise DataError('something went wrong during %s initialization' % type(self))


        def __add__(self, other):
            """
            Overloading the addition operator for magnetic types

            Args:
                other: magnetic object to be added
            Raises:
                DataError: if other is not a magnetic object
            Returns:
                sum of caller and other values (self+other)
            """

            if isinstance(other, type(self)):
                # always create new magnetic, since otherwise c = a + b changes a as well!
                M = fields.magnetic(int(np.size(self.values)/3))
                M.values = self.values + other.values
                return M
            else:
                raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))

        def __sub__(self, other):
            """
            Overloading the subrtaction operator for magnetic types

            Args:
                other: magnetic object to be subtracted
            Raises:
                DataError: if other is not a magnetic object
            Returns:
                difference of caller and other values (self-other)
            """

            if isinstance(other, type(self)):
                # always create new magnetic, since otherwise c = a + b changes a as well!
                M = fields.magnetic(int(np.size(self.values)/3))
                M.values = self.values - other.values
                return M
            else:
                raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))


        def __rmul__(self, other):
            """
            Overloading the right multiply by factor operator for magnetic types

            Args:
                other: float factor
            Raises:
                DataError: is other is not a float
            Returns:
                electric object, original values scaled by factor
            """

            if isinstance(other, float):
                # create new magnetic, no specific interpretation of float factor
                M = fields.magnetic(int(np.size(self.values)/3))
                M.values = self.values*other
                return M
            else:
                raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


    def __init__(self,init=None,vals=(None,None)):
        """
        Initialization routine

        Args:
            init: can either be a number or another fields object
            vals: initial tuple of values for electric and magnetic (default: (None,None))
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another fields object, do a deepcopy (init by copy)
        if isinstance(init,type(self)):
            self.elec = fields.electric(init.elec)
            self.magn = fields.magnetic(init.magn)
        # if init is a number, create fields object and pick the corresponding initial values
        elif isinstance(init,int):
            self.elec = fields.electric(init,val=vals[0])
            self.magn = fields.magnetic(init,val=vals[1])
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for fields types

        Args:
            other: fields object to be added
        Raises:
            DataError: if other is not a fields object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new fields, since otherwise c = a - b changes a as well!
            p = fields(int(np.size(self.elec.values)/3))
            p.elec = self.elec + other.elec
            p.magn = self.magn + other.magn
            return p
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for fields types

        Args:
            other: fields object to be subtracted
        Raises:
            DataError: if other is not a fields object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new fields, since otherwise c = a - b changes a as well!
            p = fields(int(np.size(self.elec.values)/3))
            p.elec = self.elec - other.elec
            p.magn = self.magn - other.magn
            return p
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))
