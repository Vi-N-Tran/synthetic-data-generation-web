"""Logging configuration for the trajectory generation pipeline"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        verbose: If True, use DEBUG level with more detailed formatting
        
    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if verbose:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Get root logger and configure
    logger = logging.getLogger('trajectory_generator')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (e.g., 'generator', 'llm_generator'). 
              If None, returns root logger.
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'trajectory_generator.{name}')
    return logging.getLogger('trajectory_generator')

