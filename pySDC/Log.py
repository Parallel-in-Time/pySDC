import logging

def setup_custom_logger(name, level=None):
    """
    Helper function to set main parameters for the logging facility

    Args:
        name: name for later reference
        level: level of logging
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
    if level == "info":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.addHandler(handler)

    return logger
