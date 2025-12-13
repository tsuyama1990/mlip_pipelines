import sys
from loguru import logger
from pathlib import Path

def setup_logging(output_dir: Path):
    """
    Configure loguru logger.
    """
    logger.remove() # Remove default handler

    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # File handler
    log_file = output_dir / "app.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )

    logger.info(f"Logging initialized. Log file: {log_file}")
