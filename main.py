import sys
import argparse
import warnings
import logging

# --- Logging & Warning Setup ---
# 1. Configure logging (console + rotating gui.log file) and install a global
#    exception hook FIRST, so that even import-time failures below are recorded.
from logging_setup import setup_logging
logger = setup_logging(level=logging.INFO)

# 2. Suppress PyTorch "FutureWarning" (noise)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

# 3. Suppress HuggingFace "Defaulting repo_id" warning
logging.getLogger("kokoro").setLevel(logging.ERROR)

# 4. Heavy imports are wrapped so a broken install (e.g. missing shiboken6)
#    is logged to gui.log instead of vanishing as a bare traceback.
try:
    from PySide6.QtWidgets import QApplication
    from ui_main import MyTTSMainWindow
except Exception as e:
    logger.critical(f"Failed to import application dependencies: {e}", exc_info=True)
    logger.critical(
        "This usually means the virtual environment is incomplete. "
        "Try reinstalling with: pip install --force-reinstall pyside6"
    )
    sys.exit(1)

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Kokoro TTS Local GUI")
    
    # We keep this one: It allows users to load custom settings files
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()

    # Start the GUI
    try:
        logger.info("Starting Kokoro TTS GUI...")
        app = QApplication(sys.argv)
        
        # Initialize Main Window with the config path
        window = MyTTSMainWindow(config_path=args.config)
        window.show()
        
        logger.info("Application started successfully.")
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Fatal error starting application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()