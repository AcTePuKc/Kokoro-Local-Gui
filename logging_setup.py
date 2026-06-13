import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Log file lives next to this module (repo root), gitignored via gui.log
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "gui.log")

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logging to write to both the console and a rotating
    log file (gui.log) in the repo directory, and install a global
    exception hook so that *uncaught* errors are recorded automatically.

    Safe to call multiple times; only the first call takes effect.
    """
    global _configured
    root = logging.getLogger()

    if not _configured:
        root.setLevel(level)

        # Console output
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

        # Rotating file output (5 files * 1 MB)
        try:
            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(file_handler)
        except Exception as exc:  # pragma: no cover - disk/permission issues
            root.warning("Could not create log file '%s': %s", LOG_FILE, exc)

        _install_excepthook()
        _configured = True

    return logging.getLogger("kokoro_gui")


def _install_excepthook() -> None:
    """Route any unhandled exception through the logging system."""

    def _handle(exc_type, exc_value, exc_traceback):
        # Let Ctrl+C behave normally
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("kokoro_gui").critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = _handle
