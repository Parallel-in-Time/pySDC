import logging


class FrozenClass(object):
    """
    Helper class to freeze a class, i.e. to avoid adding more attributes

    Attributes:
        __isfrozen: Flag to freeze a class
    """

    attrs = []

    __isfrozen = False

    def __setattr__(self, key, value):
        """
        Function called when setting attributes

        Args:
            key: the attribute
            value: the value
        """

        # check if attribute exists and if class is frozen
        if self.__isfrozen and not (key in self.attrs or hasattr(self, key)):
            raise TypeError(f'{type(self).__name__!r} is a frozen class, cannot add attribute {key!r}')

        object.__setattr__(self, key, value)

    def __getattr__(self, key):
        """
        This is needed in case the variables have not been initialized after adding.
        """
        if key in self.attrs:
            return None
        else:
            super().__getattribute__(key)

    @classmethod
    def add_attr(cls, key, raise_error_if_exists=False):
        """
        Add a key to the allowed attributes of this class.

        Args:
            key (str): The key to add
            raise_error_if_exists (bool): Raise an error if the attribute already exists in the class
        """
        logger = logging.getLogger(cls.__name__)
        if key in cls.attrs:
            if raise_error_if_exists:
                raise TypeError(f'Attribute {key!r} already exists in {cls.__name__}!')
            else:
                logger.debug(f'Skip adding attribute {key!r} because it already exists in {cls.__name__}!')
        else:
            cls.attrs += [key]
            logger.debug(f'Added attribute {key!r} to {cls.__name__}')

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

    def __dir__(self):
        """
        My hope is that some editors can use this for dynamic autocompletion.
        """
        return super().__dir__() + self.attrs
