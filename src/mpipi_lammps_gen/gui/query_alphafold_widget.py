import logging
import queue
from pathlib import Path

import polars as pl
from PySide6.QtCore import QSize, Signal, Slot
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from mpipi_lammps_gen.alpha_fold_query import (
    AlphaFoldQueryResult,
    query_alphafold,
)
from mpipi_lammps_gen.gui.utils.widgets import DataFrameColumnSelect, PlayPauseButton
from mpipi_lammps_gen.gui.utils.worker_thread import PlayPauseThread

from . import step_widget

logger = logging.getLogger(__name__)


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

            try:
                if file_path.suffix.lower() in [".csv", ".txt"]:
                    self.df = pl.read_csv(file_path)
                elif file_path.suffix.lower() == ".parquet":
                    self.df = pl.read_parquet(file_path)
                else:
                    msg = QMessageBox(parent=self)
                    msg.setIcon(QMessageBox.Icon.Critical)
                    msg.setWindowTitle("Incompatible file ending")
                    msg.setText(f"The file {file_path} has an unsupported suffix.")
                    msg.setInformativeText("Supported suffixes: [.txt, .csv, .parquet]")
                    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msg.exec()
                    return
            except Exception:
                logger.exception(f"Exception when trying to parse {file_path}")

            col_select_dialog = DataFrameColumnSelect(
                self.df,
                msg="Please select the column that corresponds to the UniProt accession",
                parent=self,
            )
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
    """Widget to input a list of accessions to be queried"""

    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()

        layout = QGridLayout()
        self.setLayout(layout)

        # Where the uniprot id's are pasted
        self.uniprot_ids_input = IDListTextEdit()
        self.uniprot_ids_input.setBaseSize(QSize(-1, 50))
        self.uniprot_ids_input.setPlaceholderText(
            "Enter Uniprot Accessions manually (separated by newlines) or drag and drop a file (csv or parquet)"
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

        self.accession_line_edit = QLineEdit()
        self.accession_line_edit.setReadOnly(True)
        layout.addRow("accession", self.accession_line_edit)

        self.http_status_line_edit = QLineEdit()
        self.http_status_line_edit.setReadOnly(True)
        layout.addRow("http_status", self.http_status_line_edit)

        self.n_residues_line_edit = QLineEdit()
        self.n_residues_line_edit.setReadOnly(True)
        layout.addRow("n_residues", self.n_residues_line_edit)

        self.sequence_line_edit = QLineEdit()
        self.sequence_line_edit.setReadOnly(True)
        layout.addRow("sequence", self.sequence_line_edit)

        self.plddts_line_edit = QLineEdit()
        self.plddts_line_edit.setReadOnly(True)
        layout.addRow("plddts", self.plddts_line_edit)

    def update_data(self, query_result: AlphaFoldQueryResult):
        self.accession_line_edit.setText(str(query_result.accession))
        self.http_status_line_edit.setText(str(query_result.http_status))

        if query_result.sequence is not None:
            n_residues = len(query_result.sequence)
        else:
            n_residues = 0

        self.n_residues_line_edit.setText(str(n_residues))

        self.sequence_line_edit.setText(str(query_result.sequence))
        self.plddts_line_edit.setText(str(query_result.plddts))


class QueryAlphaFoldWidget(step_widget.StepWidget):
    result_ready = Signal(object)  # emits AlphaFoldQueryResult
    step_done = Signal()  # emits after each attempt (success or fail)

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

        # Layout with control buttons and progress bar
        sublayout = QHBoxLayout()
        layout.addLayout(sublayout)

        self.query_button = PlayPauseButton()
        self.query_button.play_pressed.connect(self.on_play_button_pressed)
        self.query_button.pause_pressed.connect(self.on_pause_button_pressed)

        sublayout.addWidget(self.query_button)

        self.query_progress_bar = QProgressBar(textVisible=True)
        sublayout.addWidget(self.query_progress_bar)

        # Queried Ids
        self.queried_ids = QListWidget()
        self.queried_ids.currentRowChanged.connect(self.update_query_display)
        layout.addWidget(self.queried_ids)

        self.results_widget = QueryDisplayWidget()
        layout.addWidget(self.results_widget)

        self.result_ready.connect(self._on_result_ready)
        self.step_done.connect(self._on_step_done)

        self._query_queue = queue.Queue()
        self._query_thread = PlayPauseThread(target=self._query_loop, daemon=True)
        self._query_thread.start()

    @Slot(object)
    def _on_result_ready(self, res: AlphaFoldQueryResult):
        self.query_results.append(res)
        # accession might be in res.accession; fall back just in case
        acc = getattr(res, "accession", None)
        self.queried_ids.addItem(str(acc) if acc is not None else "(unknown)")

    @Slot()
    def _on_step_done(self):
        self.query_progress_bar.setValue(self.query_progress_bar.value() + 1)

    def _query_loop(self):
        """This function runs in a background thread and continuously queries for ids which get pushed to the queue"""
        try:
            accession = self._query_queue.get(timeout=0.5)
            try:
                res = query_alphafold(
                    accession, timeout=self.timeout, retries=self.retries
                )
                self.result_ready.emit(res)
            except Exception as e:
                logger.exception(e)
            finally:
                self.step_done.emit()
        except queue.Empty:
            ...

    @Slot()
    def on_pause_button_pressed(self):
        logger.info("Pausing query")
        self._query_thread.pause()

    @Slot()
    def on_play_button_pressed(self):
        logger.info("Running alpha fold query")

        if self._query_queue.empty():
            self.query_progress_bar.setValue(0)
            # Fill the query queue
            [self._query_queue.put(cur_id) for cur_id in self.id_list_widget.ids]

        self._query_thread.resume()
        self.query_progress_bar.setRange(0, len(self.id_list_widget.ids))

    def update_query_display(self, idx: int):
        self.results_widget.update_data(self.query_results[idx])
