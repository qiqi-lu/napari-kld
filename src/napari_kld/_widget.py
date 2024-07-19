import napari
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_kld.widgets import (
    WidgetKLDeconvPredict,
    WidgetKLDeconvTrain,
    WidgetRLDeconvButterworth,
    WidgetRLDeconvGaussian,
    WidgetRLDeconvTraditional,
    WidgetRLDeconvWB,
    WorkerRLDeconvButterworth,
    WorkerRLDeconvGaussianl,
    WorkerRLDeconvTraditional,
    WorkerRLDeconvWB,
)
from napari_kld.widgets_small import (
    FileSelectWidget,
    GaussianWidget,
    LogBox,
    ProgressObserver,
)


class RLDworker(QObject):
    finish_signal = Signal()

    def __init__(self):
        super().__init__()
        self._piplines = {}  # collect piplines of all methods
        self._current_method = ""

    def add_pipline(self, label, pipline_widget, pipline_worker):
        self._piplines.update(
            {label: {"widget": pipline_widget, "worker": pipline_worker}}
        )

    def set_method(self, name):
        self._current_method = name

    def set_image(self, image):
        worker = self._piplines[self._current_method]["worker"]
        worker.set_image(image)

    def set_psf_path(self, psf_path):
        worker = self._piplines[self._current_method]["worker"]
        worker.set_psf_path(psf_path)

    def run(self):
        print("worker run ...")
        worker = self._piplines[self._current_method]["worker"]
        worker.run()
        print("finish")
        self.finish_signal.emit()

    def current_worker(self):
        return self._piplines[self._current_method]["worker"]


class RLDwidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.events.inserted.connect(self._on_change_layer)
        self.viewer.layers.events.removed.connect(self._on_change_layer)

        self.thread = QThread()
        self._worker = RLDworker()
        self._observer = ProgressObserver()
        self._widgets = {}

        self.setLayout(QGridLayout())

        # ----------------------------------------------------------------------
        # head widget (including input, PSF, and method selection)
        head_widget = QWidget()
        head_layout = QGridLayout()
        head_widget.setLayout(head_layout)
        head_layout.setContentsMargins(0, 0, 0, 0)

        # RAW data box
        head_layout.addWidget(QLabel("Input RAW data"), 0, 0)
        self.input_raw_data_box = QComboBox()
        head_layout.addWidget(self.input_raw_data_box, 0, 1)

        # PSF box
        psf_widget = QWidget()
        psf_layout = QGridLayout()
        psf_layout.setContentsMargins(0, 0, 0, 0)

        psf_layout.addWidget(QLabel("PSF"), 0, 0)
        self.psf_mode_box = QComboBox()
        self.psf_mode_box.addItems(["Gaussian", "File", "Blind"])
        self.psf_mode_box.setCurrentText("File")
        self.psf_mode_box.currentTextChanged.connect(self._on_mode_psf_change)
        psf_layout.addWidget(self.psf_mode_box, 0, 1)

        self.psf_select = FileSelectWidget()
        psf_layout.addWidget(self.psf_select, 1, 0, 1, 2)

        self.gauss_widget = GaussianWidget()
        psf_layout.addWidget(self.gauss_widget, 2, 0, 1, 2)

        psf_widget.setLayout(psf_layout)
        head_layout.addWidget(psf_widget, 1, 0, 1, 2)

        self._on_mode_psf_change("File")

        # method
        head_layout.addWidget(QLabel("Method"), 2, 0)
        self.method_box = QComboBox()
        head_layout.addWidget(self.method_box, 2, 1)
        self.method_box.currentTextChanged.connect(self._on_change_method)

        self.layout().addWidget(head_widget, 0, 0, 1, 2)

        # ----------------------------------------------------------------------
        # method parameters
        parameters_widget = QWidget()
        self.parameters_layout = QVBoxLayout()
        self.parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_widget.setLayout(self.parameters_layout)
        self.layout().addWidget(parameters_widget, 1, 0, 1, 2)

        # ----------------------------------------------------------------------
        # run button
        self.run_btn = QPushButton("run")
        self.layout().addWidget(self.run_btn, 2, 0, 1, 2)
        self.run_btn.clicked.connect(self._on_click_run)

        # progress bar
        self.progress_bar = QProgressBar()
        self.layout().addWidget(self.progress_bar, 3, 0, 1, 2)

        # ----------------------------------------------------------------------
        # load piplines
        rld_trad_widget = WidgetRLDeconvTraditional(self.progress_bar)
        rld_trad_worker = WorkerRLDeconvTraditional(
            self.viewer, rld_trad_widget, self._observer
        )
        self.add_pipline("Traditional", rld_trad_widget, rld_trad_worker)

        rld_gaus_widget = WidgetRLDeconvGaussian(self.progress_bar)
        rld_gaus_worker = WorkerRLDeconvGaussianl(
            self.viewer, rld_gaus_widget, self._observer
        )
        self.add_pipline("Gaussian", rld_gaus_widget, rld_gaus_worker)

        rld_butt_widget = WidgetRLDeconvButterworth(self.progress_bar)
        rld_butt_worker = WorkerRLDeconvButterworth(
            self.viewer, rld_butt_widget, self._observer
        )
        self.add_pipline("Butterworth", rld_butt_widget, rld_butt_worker)

        rld_wb_widget = WidgetRLDeconvWB(self.progress_bar)
        rld_wb_worker = WorkerRLDeconvWB(
            self.viewer, rld_wb_widget, self._observer
        )
        self.add_pipline("WB", rld_wb_widget, rld_wb_worker)

        # ----------------------------------------------------------------------
        # init the view
        self._on_change_layer()
        self._on_change_method(self.method_box.currentText())

        # ----------------------------------------------------------------------
        # connect thread
        self._worker.moveToThread(self.thread)
        self.thread.started.connect(self._worker.run)
        self._worker.finish_signal.connect(self.thread.quit)
        self._worker.finish_signal.connect(self.set_outputs)

        self._observer.progress_signal.connect(self._on_progress)

    def set_outputs(self):
        worker = self._worker.current_worker()
        worker.set_outputs()

    def _on_click_run(self):
        print("run")
        # widget = self._widgets[self.method_box.currentIndex()]
        self._worker.set_method(self.method_box.currentText())
        self._worker.set_image(
            self.viewer.layers[self.input_raw_data_box.currentText()].data
        )
        self._worker.set_psf_path(self.psf_select.path_edit.text())
        self.thread.start()

    def _on_change_layer(self):
        print("layer change.")
        self.input_raw_data_box.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.input_raw_data_box.addItem(layer.name)
        if self.input_raw_data_box.count() < 1:
            self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(True)

    def _on_change_method(self, method_name):
        print(f"method change to {method_name}")
        items = (
            self.parameters_layout.itemAt(i)
            for i in range(self.parameters_layout.count())
        )

        for w in items:
            if w.widget().label == method_name:
                w.widget().setVisible(True)
            else:
                w.widget().setVisible(False)

        self.progress_bar.setValue(0)

    def add_pipline(self, label, pipline_widget, pipline_worker):
        self._worker.add_pipline(label, pipline_widget, pipline_worker)
        self._widgets.update({label: pipline_widget})
        self.parameters_layout.addWidget(pipline_widget)
        self.method_box.addItem(label)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_mode_psf_change(self, mode):
        if mode == "File":
            self.psf_select.setVisible(True)
            self.gauss_widget.setVisible(False)
        if mode == "Gaussian":
            self.psf_select.setVisible(False)
            self.gauss_widget.setVisible(True)
        if mode == "Blind":
            self.psf_select.setVisible(False)
            self.gauss_widget.setVisible(False)


class KLDworker(QObject):
    finish_signal = Signal()

    def __init__(self):
        super().__init__()

    def run(self):
        print("worker run ...")
        print("finish")
        self.finish_signal.emit()


class KLDwidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.thread = QThread()
        self._worker = RLDworker()
        self._observer = ProgressObserver()
        self._widgets = {}

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        # ----------------------------------------------------------------------
        logger = LogBox()
        logger.set_text("log:")
        self.layout().addWidget(WidgetKLDeconvTrain(logger=logger))
        self.layout().addWidget(WidgetKLDeconvPredict(viewer))
        self.layout().addWidget(logger)


if __name__ == "__main__":
    viewer = napari.Viewer()

    # dock, widget = viewer.window.add_plugin_dock_widget(
    #     "napari-kld", "RL Deconvolution"
    # )

    dock, widget = viewer.window.add_plugin_dock_widget(
        "napari-kld", "KL Deconvolution"
    )

    napari.run()
