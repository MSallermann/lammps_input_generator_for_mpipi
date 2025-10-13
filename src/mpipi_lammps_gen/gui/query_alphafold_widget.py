import logging
import queue
import threading
import time
from pathlib import Path

import polars as pl
from PySide6.QtCore import QSize
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mpipi_lammps_gen.alpha_fold_query import (
    AlphaFoldQueryResult,
    query_alphafold,
)

from . import step_widget

logger = logging.getLogger(__name__)


class DataFrameColumnSelect(QDialog):
    def __init__(self, df: pl.DataFrame, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.df = df
        self.setWindowTitle("Select column")

        layout = QVBoxLayout(self)

        # Instruction text
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("Please select the column with the UniProt IDs.")
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


class IDListTextEdit(QPlainTextEdit):
    """The widget to enter the IDs, which are to be queried"""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dropEvent(self, event: QDropEvent):  # noqa: N802
        mime = event.mimeData()

        # If files were dropped
        if mime.hasUrls():
            file_path = Path(next(url.toLocalFile() for url in mime.urls()))

            logger.info(f"Opening {file_path}")

            if file_path.suffix.lower() in [".csv", ".txt"]:
                self.df = pl.read_csv(file_path)
            elif file_path.suffix.lower() == ".parquet":
                self.df = pl.read_parquet(file_path)
            else:
                msg = QMessageBox(parent=self)
                msg.setIcon(QMessageBox.Icon.Critical)  # 👈 sets the error icon
                msg.setWindowTitle("Incompatible file ending")
                msg.setText(f"The file {file_path} has an unsupported suffix.")
                msg.setInformativeText("Supported suffixes: [.txt, .csv, .parquet]")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
                return

            col_select_dialog = DataFrameColumnSelect(self.df, parent=self)
            if col_select_dialog.exec():
                column = col_select_dialog.selected_column()
                if column is not None:
                    logger.info(f"User selected column: {column}")
                    self.clear()
                    [self.appendPlainText(s) for s in self.df[column]]
                else:
                    logger.info("No column selected.")
            else:
                logger.info("User canceled.")

        # If text was dropped
        elif mime.hasText():
            text = mime.text()
            self.setPlainText(text)

        event.acceptProposedAction()


class IDListWidget(QWidget):
    """Widget which ties together the input of"""

    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()

        layout = QGridLayout()
        self.setLayout(layout)

        # Where the uniprot id's are pasted
        self.uniprot_ids_input = IDListTextEdit()
        self.uniprot_ids_input.setBaseSize(QSize(-1, 50))
        self.uniprot_ids_input.setPlaceholderText(
            "Enter Uniprot IDs manually (separated by newlines) or drag and drop a file (csv or parquet)"
        )
        # self.uniprot_ids_input.(self.handle_drop_event)
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
    """Display the results of a single query"""

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

        # Where the uniprot id's are displayed
        self.id_list_widget = IDListWidget()
        layout.addWidget(self.id_list_widget)

        # Button to start the query
        self.query_button = QPushButton("Run Query")
        self.query_button.pressed.connect(self.on_query_button_pressed)
        layout.addWidget(self.query_button)

        # Queried Ids
        self.queried_ids = QListWidget()
        self.queried_ids.currentRowChanged.connect(self.update_query_display)
        layout.addWidget(self.queried_ids)

        self.results_widget = QueryDisplayWidget()
        layout.addWidget(self.results_widget)

        self._query_queue = queue.Queue()
        self._query_thread = threading.Thread(target=self._query_loop, daemon=True)
        self._query_thread.start()

    def _query_loop(self):
        """This function runs in a background thread and continuously queries for ids which get pushed to the queue"""
        while True:
            while not self._query_queue.empty():
                accession = self._query_queue.get()

                try:
                    res = query_alphafold(
                        accession, timeout=self.timeout, retries=self.retries
                    )
                except Exception as e:
                    logger.exception(e)
                    continue

                self.query_results.append(res)
                self.queried_ids.addItem(accession)

            time.sleep(0.5)

    def on_query_button_pressed(self):
        logger.info("Running alpha fold query")

        # Clear results
        self.queried_ids.clear()
        self.query_results.clear()

        # Empty the query queue
        self._query_queue = queue.Queue()

        # Fill the query queue
        [self._query_queue.put(cur_id) for cur_id in self.id_list_widget.ids]

    def update_query_display(self, idx: int):
        self.results_widget.update_data(self.query_results[idx])
