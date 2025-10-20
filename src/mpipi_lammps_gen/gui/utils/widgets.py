import enum

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QPushButton, QStyle


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
