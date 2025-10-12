import sys
from pathlib import Path

import tomllib
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QWidget,
)

from . import navigator_widget, pipeline_widget

DEFAULT_SETTINGS_PATH = Path(__file__).parent / "default_settings.toml"
with DEFAULT_SETTINGS_PATH.open("rb") as f:
    DEFAULT_SETTINGS = tomllib.load(f)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(DEFAULT_SETTINGS.get("MainWindow.title", ""))

        pipeline = pipeline_widget.PipelineWidget()
        navigator = navigator_widget.NavigatorWidget(pipeline_widget=pipeline)

        central = QWidget()

        central = QSplitter()
        central.addWidget(navigator)
        central.addWidget(pipeline)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 4)

        self.setCentralWidget(central)

        self.act_open = QAction(QIcon.fromTheme("document-open"), "Open…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.setStatusTip("Open a text file")
        self.act_open.triggered.connect(self.on_open)

        self.act_exit = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        self.act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_exit.setStatusTip("Exit the application")
        self.act_exit.triggered.connect(self.close)

        self.menuBar().addAction(self.act_open)
        self.menuBar().addAction(self.act_exit)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def on_open(self):
        print("lolol")


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
