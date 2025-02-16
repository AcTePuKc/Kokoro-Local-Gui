# main.py - Version 28b (Debugging Logs)

import sys
import argparse
from PySide6.QtWidgets import QApplication

# Import the PySide UI
from ui_main import MyTTSMainWindow
import logging # Import logging - Version 28b

# Configure logging - Version 28b
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="MyTTS Application")
    parser.add_argument('--train', action='store_true', help='Run training instead of GUI.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file for TTS inference.')
    parser.add_argument('--train_config', type=str, default='train_config.yaml', help='Path to config file for training.')
    args = parser.parse_args()

    if args.train:
        # Import and run training mode
        from train import run_training
        run_training(args.train_config)
    else:
        # Start the GUI for TTS
        logging.info("Starting MyTTS GUI application") # Log GUI start - Version 28b
        app = QApplication(sys.argv)
        window = MyTTSMainWindow(config_path=args.config)
        window.show()
        logging.info("GUI application main window shown") # Log main window shown - Version 28b
        sys.exit(app.exec())

if __name__ == "__main__":
    main()