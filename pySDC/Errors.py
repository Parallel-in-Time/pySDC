
class DataError(Exception):
    """
    Error Class handling/indicating problems with data types
    """
    pass


class ParameterError(Exception):
    """
    Error Class handling/indicating problems with parameters (mostly within dictionaries)
    """
    pass


class UnlockError(Exception):
    """
    Error class handling/indicating unlocked levels
    """
    pass


class CollocationError(Exception):
    """
    Error class handling/indicating problems with the collocation
    """
    pass


class TransferError(Exception):
    """
    Error class handling/indicating problems with the transfer processes
    """
    pass


class CommunicationError(Exception):
    """
    Error class handling/indicating problems with the communication
    """
    pass


class ControllerError(Exception):
    """
    Error class handling/indicating problems with the controller
    """
    pass
