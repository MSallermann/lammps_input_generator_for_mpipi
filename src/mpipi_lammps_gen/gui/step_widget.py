import enum

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


class StepState(enum.Enum):
    PENDING = 0
    READY = 1
    RUNNING = 2
    DONE = 3
    ERROR = 4
    SKIPPED = 5


class StepWidget(QWidget):
    finished = Signal(bool, str)  # ok, message

    def __init__(self, title: str, description: str):
        super().__init__()

        self._title = title
        self._description = description

        # layout = QVBoxLayout()
        # self.setLayout(layout)
        # layout.addWidget(QLabel(self.title()))

    def title(self) -> str:
        return self._title

    def description(self) -> str:
        return self._description

    def is_ready(self) -> bool:
        return True

    def explain_not_ready(self) -> str:
        return "Missing inputs."

    def run(self, start_params: dict): ...

    def _done(self, ok: bool, msg: str):
        self.setDisabled(False)
        self.finished.emit(ok, msg)
