import logging
from PySide6.QtWidgets import QMessageBox
from typing import Optional 
logger = logging.getLogger(__name__)

def show_error(parent, message: str, title: str = "Critical Error", exception: Optional[Exception] = None):
    """
    Displays a Critical Error popup and logs it to the file.
    """
    if exception:
        logger.error(f"{message} | Details: {exception}", exc_info=True)
    else:
        logger.error(message)

    if parent:
        QMessageBox.critical(parent, title, message)

def show_warning(parent, message: str, title: str = "Warning"):
    """Displays a Warning and logs it."""
    logger.warning(f"UI Warning: {message}")
    if parent:
        QMessageBox.warning(parent, title, message)

def show_info(parent, message: str, title: str = "Information"):
    """Displays an Info dialog."""
    logger.info(f"UI Info: {message}")
    if parent:
        QMessageBox.information(parent, title, message)
