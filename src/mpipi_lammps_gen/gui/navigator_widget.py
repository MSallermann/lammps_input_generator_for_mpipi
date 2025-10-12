from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
)

from . import pipeline_widget


class NavigatorWidget(QListWidget):
    def __init__(self, pipeline_widget: pipeline_widget.PipelineWidget):
        super().__init__()

        self.pipeline_widget = pipeline_widget
        self.setAlternatingRowColors(True)
        self.setIconSize(QSize(18, 18))
        self.setSpacing(4)
        self.setUniformItemSizes(True)

        for page in pipeline_widget.pages:
            item = QListWidgetItem(page.title())
            item.setToolTip(page.description())
            self.addItem(item)

        self.currentRowChanged.connect(self.select)

        self.setStyleSheet(
            """
QListWidget {
  border: none;
  background: #f6f7f9;
}
QListWidget::item {
  padding: 8px 10px;
  border-radius: 6px;
}
QListWidget::item:hover {
  background: #e9ecf1;
}
QListWidget::item:selected {
  background: #2b6cb0;     /* selection fill */
  color: white;            /* selection text */
}
"""
        )

    def select(self, idx: int):
        self.pipeline_widget.setCurrentIndex(idx)
