import logging
import os
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
from typing import Callable, Any

def setup_logger(name: str, log_file: str, level: int = logging.INFO, max_size: int = 5 * 1024 * 1024, backup_count: int = 3) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).
        max_size (int): Maximum size of log file before rotation in bytes (default: 5MB).
        backup_count (int): Number of backup files to keep (default: 3).

    Returns:
        logging.Logger: Configured logger object.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def log_execution_time(logger: logging.Logger) -> Callable:
    """
    Decorator to log the execution time of a function.

    Args:
        logger (logging.Logger): Logger object to use for logging.

    Returns:
        Callable: Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator

class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom LoggerAdapter to add context information to log messages.
    """
    def process(self, msg: str, kwargs: dict) -> tuple:
        return f'[{self.extra["context"]}] {msg}', kwargs

def get_context_logger(logger: logging.Logger, context: str) -> LoggerAdapter:
    """
    Get a logger adapter with added context.

    Args:
        logger (logging.Logger): Base logger object.
        context (str): Context string to add to log messages.

    Returns:
        LoggerAdapter: Logger adapter with context.
    """
    return LoggerAdapter(logger, {'context': context})

def log_exception(logger: logging.Logger) -> Callable:
    """
    Decorator to log exceptions raised by a function.

    Args:
        logger (logging.Logger): Logger object to use for logging.

    Returns:
        Callable: Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# Usage example:
# logger = setup_logger('my_logger', 'logs/my_app.log')
# context_logger = get_context_logger(logger, 'DataProcessing')
# 
# @log_execution_time(logger)
# @log_exception(logger)
# def my_function():
#     # Function implementation
#     pass
# 
# context_logger.info("Starting data processing")
# my_function()
# context_logger.info("Data processing completed")