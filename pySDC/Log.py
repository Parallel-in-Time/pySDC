import logging
import os

def setup_custom_logger(level=None, log_to_file=None):
    """
    Helper function to set main parameters for the logging facility

    Args:
        level: level of logging (int)
    """

    assert type(level) is int

    # specify formats and handlers
    if log_to_file:
        file_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')
        filename = 'run_pid'+str(os.getpid())+'.log'
        if os.path.isfile(filename):
            file_handler = logging.FileHandler(filename,mode='a')
        else:
            file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setFormatter(file_formatter)

    std_formatter = logging.Formatter(fmt='%(name)s - %(levelname)s: %(message)s')
    std_handler = logging.StreamHandler()
    std_handler.setFormatter(std_formatter)

    # instantiate logger
    logger = logging.getLogger('')

    # remove handlers from previous calls to controller
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)
    logger.addHandler(std_handler)
    if log_to_file:
        logger.addHandler(file_handler)
