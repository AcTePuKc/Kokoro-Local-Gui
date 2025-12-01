import sys
import argparse
import warnings
import logging
from PySide6.QtWidgets import QApplication

# Import the PySide UI
from ui_main import MyTTSMainWindow

# --- Logging & Warning Setup ---
# 1. Basic Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Suppress PyTorch "FutureWarning" (noise)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

# 3. Suppress HuggingFace "Defaulting repo_id" warning
logging.getLogger("kokoro").setLevel(logging.ERROR)

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