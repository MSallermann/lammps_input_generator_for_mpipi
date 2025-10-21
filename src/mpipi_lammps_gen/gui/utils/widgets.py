import enum

import polars as pl
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class PlayPauseState(enum.Enum):
    pause = 0
    play = 1


class PlayPauseButton(QPushButton):
    play_pressed = Signal()
    pause_pressed = Signal()

    def __init__(
        self, initial_state: PlayPauseState = PlayPauseState.play, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.setAutoDefault(False)

        s = self.style()
        self._play_icon = s.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._pause_icon = s.standardIcon(QStyle.StandardPixmap.SP_MediaPause)

        self.setIcon(self._play_icon)

        # reflect initial state
        self.setChecked(initial_state == PlayPauseState.pause)
        self._apply_icon(self.isChecked())

        self.toggled.connect(self._on_toggled)

    @Slot(bool)
    def _on_toggled(self, playing: bool):
        # when checked -> pause icon (meaning currently playing)
        self._apply_icon(playing)
        (self.play_pressed if playing else self.pause_pressed).emit()

    def _apply_icon(self, playing: bool):
        self.setIcon(self._pause_icon if playing else self._play_icon)


class DataFrameColumnSelect(QDialog):
    def __init__(
        self, df: pl.DataFrame, msg: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent=parent)
        self.df = df
        self.setWindowTitle("Select column")

        layout = QVBoxLayout(self)

        # Instruction text
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(msg)
        layout.addWidget(text)

        # Combo box with column names
        self.combobox = QComboBox()
        self.combobox.addItems(self.df.columns)
        layout.addWidget(self.combobox)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_column(self) -> str | None:
        """Return the selected column name, or None if cancelled."""
        if self.result() == QDialog.DialogCode.Accepted:
            return self.combobox.currentText()
        return None
