import logging

def setup_custom_logger(level=None):
    """
    Helper function to set main parameters for the logging facility

    Args:
        level: level of logging (int)
    """

    assert type(level) is int


    # speficy format
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)



    # instantiate logger
    logger = logging.getLogger('')

    # remove handlers from previous calls to controller
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)
    logger.addHandler(handler)

