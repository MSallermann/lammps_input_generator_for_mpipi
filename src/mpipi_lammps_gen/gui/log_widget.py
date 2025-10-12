from logging import Handler, LogRecord

from PySide6.QtWidgets import QTextEdit


class LogWidget(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)


class LogWidgetHandler(Handler):
    def __init__(self, log_widget: LogWidget, level: int | str = 0) -> None:
        super().__init__(level)
        self.log_widget = log_widget

    def emit(self, record: LogRecord):
        self.log_widget.append(record.getMessage())
