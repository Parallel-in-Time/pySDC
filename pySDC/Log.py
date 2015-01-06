import logging

def setup_custom_logger(name):
    """
    Helper function to set main parameters for the logging facility

    Args:
        name: name for later reference
    Returns:
        logger to work with
    """

    # speficy format
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # instantiate logger
    logger = logging.getLogger(name)
    # set level of logging and join with format
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
