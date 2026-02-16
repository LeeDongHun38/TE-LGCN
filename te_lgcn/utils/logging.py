"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(name='te_lgcn', log_file=None, level=logging.INFO):
    """
    Setup logger with console and file handlers.

    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level: Logging level

    Returns:
        logging.Logger: Configured logger

    Example:
        >>> logger = setup_logger('te_lgcn', 'logs/training.log')
        >>> logger.info('Training started')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
