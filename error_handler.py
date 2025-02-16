from PySide6.QtWidgets import QMessageBox

def show_error(parent, message, title="Error"):
    """
    Display an error message in a QMessageBox.
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec()
