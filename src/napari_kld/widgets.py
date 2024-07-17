# the widgets for each methods
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QSpinBox, QVBoxLayout, QWidget

from . import methods


# traditional RLD
class WidgetRLDeconvTraditional(QWidget):
    def __init__(self):
        super().__init__()
        self.label = "Traditional"

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.iteration_box = QSpinBox()
        layout.addWidget(self.iteration_box)


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
        self._out = methods.rl_deconv(self.image)

    def set_outputs(self):
        self.viewer.add_image(self._out, name="RLD Traditional")
