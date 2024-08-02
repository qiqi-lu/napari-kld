import napari
import qtpy.QtCore
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari_kld.baseww import (
    FileSelectWidget,
    LogBox,
    WidgetBase,
)
from napari_kld.widgets import (
    WidgetKLDeconvPredict,
    WidgetKLDeconvSimulation,
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


class RLDwidget(WidgetBase):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.events.inserted.connect(self._on_change_layer)
        self.viewer.layers.events.removed.connect(self._on_change_layer)

        self._worker = RLDworker()
        self._widgets = {}

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        # RAW data box
        grid_layout.addWidget(QLabel("Input RAW data"), 0, 0, 1, 1)
        self.input_raw_data_box = QComboBox()
        grid_layout.addWidget(self.input_raw_data_box, 0, 1, 1, 1)

        # ----------------------------------------------------------------------
        # PSF box
        grid_layout.addWidget(QLabel("PSF"), 1, 0, 1, 1)
        self.paf_path_box = FileSelectWidget()
        grid_layout.addWidget(self.paf_path_box, 1, 1, 1, 1)

        # ----------------------------------------------------------------------
        # method
        grid_layout.addWidget(QLabel("Method"), 2, 0, 1, 1)
        self.method_box = QComboBox()
        grid_layout.addWidget(self.method_box, 2, 1, 1, 1)
        self.method_box.currentTextChanged.connect(self._on_change_method)

        # ----------------------------------------------------------------------
        # method parameters
        parameters_widget = QWidget()
        self.parameters_layout = QVBoxLayout()
        self.parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_widget.setLayout(self.parameters_layout)
        grid_layout.addWidget(parameters_widget, 3, 0, 1, 2)

        # ----------------------------------------------------------------------
        self.layout().addWidget(self.run_btn, 4, 0, 1, 2)
        self.layout().addWidget(self.progress_bar, 5, 0, 1, 2)

        grid_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)

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
        self.reconnect()
        self._worker.finish_signal.connect(self.set_outputs)

    def set_outputs(self):
        worker = self._worker.current_worker()
        worker.set_outputs()

    def _on_click_run(self):
        print("run")
        # widget = self._widgets[self.method_box.currentIndex()]
        self._worker.set_method(self.method_box.currentText())
        self._worker.set_image(
            self.viewer.layers[self.input_raw_data_box.currentText()],
        )
        self._worker.set_psf_path(self.paf_path_box.get_path())
        self._thread.start()

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
                self.progress_bar.setMaximum(w.widget().iteration_box.value())

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


class KLDwidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self._widgets = {}
        logger = LogBox()
        # ----------------------------------------------------------------------
        page_widget_train = QWidget()
        page_layout_train = QVBoxLayout()
        page_widget_train.setLayout(page_layout_train)
        page_layout_train.addWidget(WidgetKLDeconvTrain(logger=logger))

        # ----------------------------------------------------------------------
        page_widget_prediction = QWidget()
        page_layout_prediction = QVBoxLayout()
        page_widget_prediction.setLayout(page_layout_prediction)
        page_layout_prediction.addWidget(WidgetKLDeconvPredict(viewer, logger))
        page_layout_prediction.addStretch()

        # ----------------------------------------------------------------------
        page_widget_simulation = QWidget()
        page_layout_simulation = QVBoxLayout()
        page_widget_simulation.setLayout(page_layout_simulation)
        page_layout_simulation.addWidget(WidgetKLDeconvSimulation(logger))
        page_layout_simulation.addStretch()

        # ----------------------------------------------------------------------
        page_widget_log = QWidget()
        page_layout_log = QVBoxLayout()
        page_widget_log.setLayout(page_layout_log)
        page_layout_log.addWidget(logger)
        page_layout_log.addStretch()

        # ----------------------------------------------------------------------
        tabwidget = QTabWidget()
        tabwidget.addTab(page_widget_train, "Training")
        tabwidget.addTab(page_widget_prediction, "Prediction")
        tabwidget.addTab(page_widget_simulation, "Simulation")
        tabwidget.addTab(page_widget_log, "Log")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabwidget)
        self.setLayout(layout)


if __name__ == "__main__":
    viewer = napari.Viewer()

    # dock, widget = viewer.window.add_plugin_dock_widget(
    #     "napari-kld", "RL Deconvolution"
    # )

    dock, widget = viewer.window.add_plugin_dock_widget(
        "napari-kld", "KL Deconvolution"
    )

    napari.run()
