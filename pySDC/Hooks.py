from pySDC.Level import level

class hooks():

    __slots__ = ('__level')

    def __init__(self):
        """
        Initialization routine
        """
        self.__level = None
        pass


    def __set_level(self,L):
        """
        Sets a reference to the current level (done in the initialization of the level)

        Args:
            L: current level
        """
        assert isinstance(L,level)
        self.__level = L


    @property
    def level(self):
        """
        Getter for the current level
        Returns:
            level
        """
        return self.__level


    def dump_iteration(self):
        """
        Default routine called after each iteration
        """
        pass


    def dump_step(self):
        """
        Default routine called after each step
        """
        pass