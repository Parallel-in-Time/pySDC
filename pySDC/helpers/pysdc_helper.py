class FrozenClass(object):
    """
    Helper class to freeze a class, i.e. to avoid adding more attributes

    Attributes:
        __isfrozen: Flag to freeze a class
    """

    __isfrozen = False

    def __setattr__(self, key, value):
        """
        Function called when setting arttributes

        Args:
            key: the attribute
            value: the value
        """

        # check if attribute exists and if class is frozen
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        """
        Function to freeze the class
        """
        self.__isfrozen = True

    def get(self, key, default=None):
        """
        Wrapper for `__dict__.get` to use when reading variables that might not exist, depending on the configuration

        Args:
            key (str): Name of the variable you wish to read
            default: Value to be returned if the variable does not exist

        Returns:
            __dict__.get(key, default)
        """
        return self.__dict__.get(key, default)
