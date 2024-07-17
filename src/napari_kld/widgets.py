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

from . import methods


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

    def set_image(self, image):
        self.image = image

    def run(self):
        num_iter = self.widget.iteration_box.value()
        self._out = methods.rl_deconv(self.image, num_iter, self.obsever)

    def set_outputs(self):
        self.viewer.add_image(self._out, name="RLD (Traditional)")


# RLD using Guassian kernel
class WidgetRLDeconvGaussian(WidgetRLDeconvTraditional):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "Gaussian"


class WorkerRLDeconvGaussianl(WorkerRLDeconvTraditional):
    def __init__(self, viewer, widget, observer):
        super().__init__(viewer, widget, observer)

    def run(self):
        num_iter = self.widget.iteration_box.value()
        self._out = methods.rl_deconv(self.image, num_iter, self.obsever)

    def set_outputs(self):
        self.viewer.add_image(self._out, name="RLD (Gaussian)")


# RLD using Butterworth kernel
class WidgetRLDeconvButterworth(WidgetRLDeconvTraditional):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "Butterworth"

        self.layout_grid.addWidget(QLabel("Alpha:"), 1, 0)
        self.alpha_box = QDoubleSpinBox()
        self.alpha_box.setDecimals(5)
        self.layout_grid.addWidget(self.alpha_box, 1, 1)


class WorkerRLDeconvButterworth(WorkerRLDeconvTraditional):
    def __init__(self, viewer, widget, observer):
        super().__init__(viewer, widget, observer)

    def run(self):
        num_iter = self.widget.iteration_box.value()
        self._out = methods.rl_deconv(self.image, num_iter, self.obsever)

    def set_outputs(self):
        self.viewer.add_image(self._out, name="RLD (Butterworth)")
