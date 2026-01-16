"""Logging configuration for UAAG2 using loguru.

This module provides a pre-configured logger for use throughout the project.
Import the logger from this module to get consistent logging across the codebase.

Example usage:
    from uaag2.logging_config import logger

    logger.info("Training started")
    logger.debug("Batch size: {}", batch_size)
    logger.warning("Low memory detected")
    logger.error("Failed to load checkpoint")
"""

import sys
from pathlib import Path

from loguru import logger

# Remove the default logger to configure our own
logger.remove()

# Console handler - INFO level and above with colors
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)


def configure_file_logging(log_dir: str | Path, level: str = "DEBUG", rotation: str = "100 MB") -> None:
    """Configure file logging to save logs to a specified directory.

    Args:
        log_dir: Directory where log files will be saved.
        level: Minimum log level for file logging. Defaults to "DEBUG".
        rotation: When to rotate the log file. Defaults to "100 MB".
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "uaag2_{time:YYYY-MM-DD}.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention="30 days",
        compression="zip",
    )


def set_log_level(level: str) -> None:
    """Set the console log level dynamically.

    Args:
        level: Log level to set (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )


__all__ = ["logger", "configure_file_logging", "set_log_level"]
