from loguru import logger
import sys

def setup_app_logger(name):
    """
    Set up a logger for the application using Loguru.

    Returns:
        A logger object configured with a default sink.
    """
    logger.remove()  # Remove default handler to avoid duplicate logs if reconfigured
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
        level="INFO"
    )
    return logger

# Example of how to use this setup function in your module:
# from utils.logger import setup_app_logger
# logger = setup_app_logger()
#
# logger.debug("This is a debug message from the application.")
# logger.info("Informational message.")
# logger.warning("A warning occurred.")
# logger.error("An error occurred.")
# logger.critical("A critical error occurred.")
