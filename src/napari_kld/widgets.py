# the widgets for each methods
import qtpy.QtCore
from qtpy.QtCore import QObject
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_kld import methods


# traditional RLD
class WidgetRLDeconvTraditional(QWidget):
    def __init__(self, progress_bar):
        super().__init__()
        self.label = "Traditional"
        self.progress_bar = progress_bar

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # parameter box
        self.params_group = QGroupBox()
        self.params_group.setTitle(self.label)

        self.layout_grid = QGridLayout()
        self.layout_grid.setContentsMargins(3, 11, 3, 11)

        self.layout_grid.addWidget(QLabel("Iterations:"), 0, 0)
        self.iteration_box = QSpinBox()
        self.iteration_box.setMinimum(1)
        self.iteration_box.setValue(30)
        self.progress_bar.setMaximum(30)
        self.iteration_box.valueChanged.connect(self._on_num_iter_change)
        self.layout_grid.addWidget(self.iteration_box, 0, 1)

        self.params_group.setLayout(self.layout_grid)
        self.layout.addWidget(self.params_group)
        self.layout.addWidget(QWidget(), 1, qtpy.QtCore.Qt.AlignTop)

        # init view
        self._on_num_iter_change()

    def _on_num_iter_change(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(self.iteration_box.value())


class WorkerRLDeconvTraditional(QObject):
    def __init__(self, viewer, widget, observer):
        super().__init__()
        self.viewer = viewer
        self.widget = widget
        self.obsever = observer

        self._out = None
        self.image = None
        self.psf_path = None

    def set_image(self, image):
        self.image = image

    def set_psf_path(self, psf_path):
        self.psf_path = psf_path

    def run(self):
        dict_params = {
            "num_iter": self.widget.iteration_box.value(),
            "psf_path": self.psf_path,
            "kernel_type": self.widget.label,
        }
        self._out = methods.rl_deconv(
            self.image, observer=self.obsever, **dict_params
        )

    def set_outputs(self):
        self.viewer.add_image(
            self._out,
            name=f"deconv_{self.widget.label.lower()}_iter_{self.widget.iteration_box.value()}",
        )


# RLD using Guassian kernel
class WidgetRLDeconvGaussian(WidgetRLDeconvTraditional):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "Gaussian"
        self.iteration_box.setValue(30)
        self.progress_bar.setMaximum(30)


class WorkerRLDeconvGaussianl(WorkerRLDeconvTraditional):
    def __init__(self, viewer, widget, observer):
        super().__init__(viewer, widget, observer)

    def run(self):
        dict_params = {
            "num_iter": self.widget.iteration_box.value(),
            "psf_path": self.psf_path,
            "kernel_type": self.widget.label,
        }
        self._out = methods.rl_deconv(
            self.image, observer=self.obsever, **dict_params
        )


# RLD using Butterworth kernel
class WidgetRLDeconvButterworth(WidgetRLDeconvTraditional):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "Butterworth"

        self.iteration_box.setValue(30)
        self.progress_bar.setMaximum(30)

        # beta box
        self.layout_grid.addWidget(QLabel("beta:"), 1, 0)
        self.beta_box = QDoubleSpinBox()
        self.beta_box.setDecimals(5)
        self.beta_box.setSingleStep(0.01)
        self.beta_box.setValue(0.01)
        self.layout_grid.addWidget(self.beta_box, 1, 1)

        # n box
        self.layout_grid.addWidget(QLabel("n:"), 2, 0)
        self.n_box = QSpinBox()
        self.n_box.setValue(10)
        self.layout_grid.addWidget(self.n_box, 2, 1)


class WorkerRLDeconvButterworth(WorkerRLDeconvTraditional):
    def __init__(self, viewer, widget, observer):
        super().__init__(viewer, widget, observer)

    def run(self):
        dict_params = {
            "num_iter": self.widget.iteration_box.value(),
            "psf_path": self.psf_path,
            "kernel_type": self.widget.label,
            "beta": self.widget.beta_box.value(),
            "n": self.widget.n_box.value(),
        }
        self._out = methods.rl_deconv(
            self.image, observer=self.obsever, **dict_params
        )

    def set_outputs(self):
        self.viewer.add_image(
            self._out,
            name=f"deconv_{self.widget.label.lower()}_iter_{self.widget.iteration_box.value()}_beta_{self.widget.beta_box.value():.5f}_n_{self.widget.n_box.value()}",
        )


# RLD using WB kernel
class WidgetRLDeconvWB(WidgetRLDeconvButterworth):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "WB"

        # alpha box
        self.layout_grid.addWidget(QLabel("alpha"), 3, 0)
        self.alpha_box = QDoubleSpinBox()
        self.alpha_box.setDecimals(5)
        self.alpha_box.setSingleStep(0.001)
        self.alpha_box.setValue(0.005)
        self.layout_grid.addWidget(self.alpha_box, 3, 1)

        self.beta_box.setValue(0.1)
        self.iteration_box.setValue(2)
        self.progress_bar.setMaximum(2)


class WorkerRLDeconvWB(WorkerRLDeconvTraditional):
    def __init__(self, viewer, widget, observer):
        super().__init__(viewer, widget, observer)

    def run(self):
        dict_params = {
            "num_iter": self.widget.iteration_box.value(),
            "psf_path": self.psf_path,
            "kernel_type": self.widget.label,
            "alpha": self.widget.alpha_box.value(),
            "beta": self.widget.beta_box.value(),
            "n": self.widget.n_box.value(),
        }
        self._out = methods.rl_deconv(
            self.image, observer=self.obsever, **dict_params
        )

    def set_outputs(self):
        self.viewer.add_image(
            self._out,
            name=f"deconv_{self.widget.label.lower()}_iter_{self.widget.iteration_box.value()}_beta_{self.widget.beta_box.value():.5f}_n_{self.widget.n_box.value()}_alpha_{self.widget.alpha_box.value():.5f}",
        )
