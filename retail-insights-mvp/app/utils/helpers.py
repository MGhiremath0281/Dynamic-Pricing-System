import logging
import os
from config import LOG_FILE

def get_logger(name):
    """
    Create and return a logger with the given name.
    Logs are written to the file specified in config.LOG_FILE.
    """
    # Make sure the log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Create or retrieve the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already added
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
