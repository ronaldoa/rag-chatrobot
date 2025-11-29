"""Utility helpers."""
import os
from pathlib import Path
from loguru import logger
import sys

def setup_logger(log_path: str = "./logs"):
    """
    Configure logging.

    Args:
        log_path: Directory for log files.

    Returns:
        logger instance
    """
    # Ensure log directory exists
    os.makedirs(log_path, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # File output
    logger.add(
        f"{log_path}/app.log",
        rotation="100 MB",      # Rotate at 100MB
        retention="10 days",    # Keep 10 days
        compression="zip",      # Compress old logs
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger

def ensure_directories(*dirs):
    """
    Ensure directories exist.

    Args:
        *dirs: Directory paths
    """
    for directory in dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

def get_file_size(filepath: str) -> str:
    """
    Return file size in human-readable form.

    Args:
        filepath: File path

    Returns:
        Formatted size string
    """
    size = Path(filepath).stat().st_size

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} PB"

def format_sources(sources: list, max_length: int = 200) -> str:
    """
    Format source metadata for display.

    Args:
        sources: List of sources
        max_length: Maximum length to show for each source

    Returns:
        Formatted source string
    """
    if not sources:
        return ""

    formatted = "\n\n---\n📚 **References:**\n"
    for i, source in enumerate(sources, 1):
        content = source.get("content", "")[:max_length]
        source_name = source.get("source", "Unknown")
        page = source.get("page")

        formatted += f"{i}. `{source_name}`"
        if page:
            formatted += f" (page {page})"
        formatted += f"\n   {content}...\n"

    return formatted
