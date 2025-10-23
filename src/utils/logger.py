import logging
import os
from src.config.settings import config

def setup_logger(name: str = 'agriagent') -> logging.Logger:
    """Set up logger with file and console handlers"""

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Global logger instance
logger = setup_logger()
