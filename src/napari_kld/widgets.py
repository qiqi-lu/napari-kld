# the widgets for each methods
from qtpy.QtWidgets import QSpinBox, QVBoxLayout, QWidget


# traditional RLD
class WidgetRLDeconvTraditional(QWidget):
    def __init__(self):
        super.__init__()
        self.label = "Taditional"

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.iteration_box = QSpinBox()
        layout.addWidget(self.iteration_box)


# class WidgetRLDeconvTraditional(QObject):
