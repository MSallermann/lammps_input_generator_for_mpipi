import logging

from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mpipi_lammps_gen.alpha_fold_query import (
    query_alphafold_bulk,
)

from . import step_widget

logger = logging.getLogger(__name__)


class IDListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Where the uniprot id's are pasted / displayed
        self.uniprot_ids_input = QPlainTextEdit()
        self.uniprot_ids_input.textChanged.connect(self.try_to_parse_ids)
        layout.addWidget(self.uniprot_ids_input)

        stats_container = QWidget()
        layout.addWidget(stats_container)

        stats_layout = QGridLayout()
        stats_container.setLayout(stats_layout)

        self.number_of_ids = QLabel("No. IDs: 0")
        stats_layout.addWidget(self.number_of_ids, 0, 0)

    def try_to_parse_ids(self):
        text = self.uniprot_ids_input.toPlainText()
        lines = text.split()
        self.ids = [line.strip() for line in lines]

        self.number_of_ids.setText(f"No. IDs: {len(self.ids)}")


class QueryAlphaFoldWidget(step_widget.StepWidget):
    def __init__(self):
        super().__init__(
            title="QueryAlphaFold",
            description="Query the alpha fold DB with a uniprot ID",
        )

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.id_list_widget = IDListWidget()
        layout.addWidget(self.id_list_widget)

        self.parse_button = QPushButton("Query")
        self.parse_button.pressed.connect(self.run_query)
        layout.addWidget(self.parse_button)

    def run_query(self):
        logger.warning("Running alpha fold query")
        logger.info("info Running alpha fold query")

        query_alphafold_bulk(accession_list=self.id_list_widget.ids)
