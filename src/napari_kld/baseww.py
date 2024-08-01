import os

from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QWidget,
)


class FileSelectWidget(QWidget):
    """allow users to select files."""

    def __init__(self, title="Select a file"):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.path_edit = QLineEdit()
        layout.addWidget(self.path_edit)

        self.btn_browse = QPushButton("Choose")
        self.btn_browse.released.connect(self._on_browse)
        layout.addWidget(self.btn_browse)

        self.title = title

        self.set_enabled(True)

    def _on_browse(self):
        init_directory = os.getcwd()
        file = QFileDialog.getOpenFileName(
            self, self.title, init_directory, "*.*"
        )
        if file != "":
            self.path_edit.setText(file[0])

    def get_path(self):
        print(self.path_edit.text())
        return self.path_edit.text()

    def set_enabled(self, enable):
        self.path_edit.setEnabled(enable)
        self.btn_browse.setEnabled(enable)


class DirectorySelectWidget(FileSelectWidget):
    def __init__(self, title="Select a folder"):
        super().__init__(title)

    def _on_browse(self):
        init_directory = os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self, self.title, init_directory
        )
        if directory != "":
            self.path_edit.setText(directory)


class GaussianWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        layout.addWidget(QLabel("Sigma"), 0, 0)
        self.sigma_box = QDoubleSpinBox()
        self.sigma_box.setDecimals(5)
        self.sigma_box.setMinimum(0)
        self.sigma_box.setSingleStep(0.01)
        layout.addWidget(self.sigma_box, 0, 1)


class ProgressObserver(QObject):
    progress_signal = Signal(int)
    notify_signal = Signal(str)

    def __init__(self):
        super().__init__()

    def progress(self, value):
        self.progress_signal.emit(value)

    def notify(self, message):
        self.notify_signal.emit(message)

    def pop_info(self, info):
        show_info(info)


class LogBox(QGroupBox):
    def __init__(self):
        super().__init__()

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        grid_layout.addWidget(self.log_box, 1, 0, 4, 4)

        self.clear_btn = QPushButton("clear")
        grid_layout.addWidget(self.clear_btn, 0, 0, 1, 1)
        self.clear_btn.clicked.connect(self.clear_text)

    def add_text(self, value):
        self.log_box.append(value)

    def clear_text(self):
        self.log_box.clear()


class LineBox(QWidget):
    def __init__(self, name):
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(name), 0, 0, 1, 1)
        self.edit_box = QLineEdit()
        layout.addWidget(self.edit_box, 0, 1, 1, 2)

    def get_text(self):
        return self.edit_box.text()

    def set_text(self, text):
        self.edit_box.setText(text)


class SpinBox(QSpinBox):
    def __init__(self, vmin, vmax, vinit):
        super().__init__()

        self.setMinimum(vmin)
        self.setMaximum(vmax)
        self.setValue(vinit)


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, vmin, vmax, vinit, decimals=8):
        super().__init__()

        self.setDecimals(decimals)
        self.setMinimum(vmin)
        self.setMaximum(vmax)
        self.setValue(vinit)


class WorkerBase(QObject):
    finish_signal = Signal()
    succeed_signal = Signal()

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer
        self.params_dict = {}

    def set_params(self, params_dict):
        self.params_dict.update(params_dict)

    def run(self):
        print("worker run ...")


class WidgetBase(QGroupBox):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self._thread = QThread()
        self._observer = ProgressObserver()
        self._worker = WorkerBase()

        self.run_btn = QPushButton("run")
        self.progress_bar = QProgressBar()

    def reconnect(self):
        self.run_btn.clicked.connect(self._on_click_run)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finish_signal.connect(self._thread.quit)

        self._observer.progress_signal.connect(self._on_progress)
        self._observer.notify_signal.connect(self._on_notify)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_notify(self, value):
        if self.logger is not None:
            self.logger.add_text(value)

    def print_params(self, params_dict):
        self._on_notify("Parameters:")
        for item in params_dict:
            self._on_notify(f"{item} : {params_dict[item]}")

    def restart(self):
        self.progress_bar.setValue(0)
        self._thread.quit()

    def _on_click_run(self):
        print("run ...")

    def enable_run(self, enable):
        self.run_btn.setEnabled(enable)
