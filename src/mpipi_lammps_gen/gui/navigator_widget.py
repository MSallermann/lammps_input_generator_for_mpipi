from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
)

from . import pipeline_widget


class NavigatorWidget(QListWidget):
    def __init__(self, pipeline_widget: pipeline_widget.PipelineWidget):
        super().__init__()

        self.pipeline_widget = pipeline_widget

        for page in pipeline_widget.pages:
            self.addItem(QListWidgetItem(page.title()))

        self.currentRowChanged.connect(self.select)

    def select(self, idx: int):
        print(idx)
        self.pipeline_widget.setCurrentIndex(idx)
