import logging

from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mpipi_lammps_gen.alpha_fold_query import AlphaFoldQueryResult, query_alphafold_bulk

from . import step_widget

logger = logging.getLogger(__name__)


class IDListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()

        layout = QGridLayout()
        self.setLayout(layout)

        # Where the uniprot id's are pasted
        self.uniprot_ids_input = QPlainTextEdit()
        self.uniprot_ids_input.textChanged.connect(self.try_to_parse_ids)
        layout.addWidget(self.uniprot_ids_input, 0, 0)

        # Where the stats for each query are displayed
        stats_container = QWidget()
        layout.addWidget(stats_container, 1, 0)

        stats_layout = QGridLayout()
        stats_container.setLayout(stats_layout)
        self.number_of_ids = QLabel("No. IDs: 0")
        stats_layout.addWidget(self.number_of_ids, 0, 0)

    def try_to_parse_ids(self):
        text = self.uniprot_ids_input.toPlainText()
        lines = text.split()

        self.ids = {line.strip() for line in lines}

        self.number_of_ids.setText(f"No. IDs: {len(self.ids)}")


class QueryDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QFormLayout()
        self.setLayout(layout)

        self.accession = QLineEdit()
        self.accession.setReadOnly(True)
        layout.addRow("accession", self.accession)

        self.http_status = QLineEdit()
        self.http_status.setReadOnly(True)
        layout.addRow("http_status", self.http_status)

        self.n_residues = QLineEdit()
        self.n_residues.setReadOnly(True)
        layout.addRow("n_residues", self.n_residues)

        self.sequence = QLineEdit()
        self.sequence.setReadOnly(True)
        layout.addRow("sequence", self.sequence)

        self.plddts = QLineEdit()
        self.plddts.setReadOnly(True)
        layout.addRow("plddts", self.plddts)

    def update_data(self, query_result: AlphaFoldQueryResult):
        self.accession.setText(str(query_result.accession))
        self.http_status.setText(str(query_result.http_status))

        if query_result.sequence is not None:
            n_residues = len(query_result.sequence)
        else:
            n_residues = 0

        self.n_residues.setText(str(n_residues))

        self.sequence.setText(str(query_result.sequence))
        self.plddts.setText(str(query_result.plddts))


class QueryAlphaFoldWidget(step_widget.StepWidget):
    def __init__(self):
        super().__init__(
            title="QueryAlphaFold",
            description="Query the alpha fold DB with a uniprot ID",
        )

        self.query_results: list[AlphaFoldQueryResult] = []
        self.timeout: int = 1
        self.retries: int = 0

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.id_list_widget = IDListWidget()
        layout.addWidget(self.id_list_widget)

        self.parse_button = QPushButton("Query")
        self.parse_button.pressed.connect(self.run_query)
        layout.addWidget(self.parse_button)

        # Where the uniprot id's are displayed
        self.parsed_ids = QListWidget()
        self.parsed_ids.currentRowChanged.connect(self.update_query_display)
        layout.addWidget(self.parsed_ids)

        self.results_widget = QueryDisplayWidget()
        layout.addWidget(self.results_widget)

    def run_query(self):
        logger.info("Running alpha fold query")

        self.parsed_ids.clear()
        self.query_results.clear()

        for q in query_alphafold_bulk(
            accession_list=self.id_list_widget.ids,
            timeout=self.timeout,
            retries=self.retries,
        ):
            self.query_results.append(q)
            if q.accession is not None:
                self.parsed_ids.addItem(q.accession)

    def update_query_display(self, idx: int):
        self.results_widget.update_data(self.query_results[idx])
