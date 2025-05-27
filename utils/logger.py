import logging

def setup_app_logger(name):
    """
    Set up a logger for the application.

    Args:
        name: The name of the logger

    Returns:
        A logger object
    """
    logger = logging.getLogger(name)
    
    # Set the default level to INFO
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Example of how to use this setup function in your module:
# logger = setup_app_logger(__name__)
#
# logger.debug("This is a debug message from the application.")
# logger.info("Informational message.")
# logger.warning("A warning occurred.")
# logger.error("An error occurred.")
# logger.critical("A critical error occurred.")
