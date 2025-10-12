from PySide6.QtWidgets import (
    QStackedWidget,
)

from . import parse_file_widget, query_alphafold_widget, step_widget


class PipelineWidget(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.pages: list[step_widget.StepWidget] = []

        self.add_page(query_alphafold_widget.QueryAlphaFoldWidget())
        self.add_page(parse_file_widget.ParseFileWidget())

        self.setCurrentIndex(0)

    def add_page(self, page: step_widget.StepWidget):
        self.addWidget(page)
        self.pages.append(page)
