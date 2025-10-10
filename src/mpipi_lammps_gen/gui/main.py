import sys
from pathlib import Path

import tomllib
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

DEFAULT_SETTINGS_PATH = Path(__file__).parent / "default_settings.toml"
with DEFAULT_SETTINGS_PATH.open("rb") as f:
    DEFAULT_SETTINGS = tomllib.load(f)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(DEFAULT_SETTINGS.get("MainWindow.title", ""))

        button = QPushButton("Press Me!")

        # Set the central widget of the Window.
        self.setCentralWidget(button)


def main():
    # You need one (and only one) QApplication instance per application.
    # Pass in sys.argv to allow command line arguments for your app.
    # If you know you won't use command line arguments QApplication([]) works too.
    app = QApplication(sys.argv)

    # Create a Qt widget, which will be our window.
    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()
