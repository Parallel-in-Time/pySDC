
class DataError(Exception):
    """
    Custom error class for data error, e.g. during initialiation of data type operations

    Attributes:
        value: a string which will contain the message provided by the user/caller
    """

    def __init__(self, value):
        """
        Initialization routine

        Args:
            value: a string which will contain the message provided by the user/caller
        """

        self.value = value

    def __str__(self):
        """
        Returns the string

        Returns
            value attribute
        """

        return self.value


