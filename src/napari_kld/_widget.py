"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""

import napari
from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtCore import QObject, QThread
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.util import img_as_float

import napari_kld.methods as methods

# import numpy as np


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def threshold_autogenerate_widget(
    img: "napari.types.ImageData",
    threshold: "float",
) -> "napari.types.LabelsData":
    return img_as_float(img) > threshold


def rl_deconv_widget(
    img: "napari.types.ImageData",
) -> "napari.types.ImageData":
    img_deconv = methods.rl_deconv(img)
    return img_deconv


# the magic_factory decorator lets us customize aspects of our widget
# we specify a widget type for the threshold parameter
# and use auto_call=True so the function is called whenever
# the value of a parameter changes
@magic_factory(
    threshold={"widget_type": "FloatSlider", "max": 1}, auto_call=True
)
def threshold_magic_widget(
    img_layer: "napari.layers.Image", threshold: "float"
) -> "napari.types.LabelsData":
    return img_as_float(img_layer.data) > threshold


# if we want even more control over our widget, we can use
# magicgui `Container`
class ImageThreshold(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._threshold_slider = create_widget(
            label="Threshold", annotation=float, widget_type="FloatSlider"
        )
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        # use magicgui widgets directly
        self._invert_checkbox = CheckBox(text="Keep pixels below threshold")

        # connect your own callbacks
        self._threshold_slider.changed.connect(self._threshold_im)
        self._invert_checkbox.changed.connect(self._threshold_im)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._image_layer_combo,
                self._threshold_slider,
                self._invert_checkbox,
            ]
        )

    def _threshold_im(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        image = img_as_float(image_layer.data)
        name = image_layer.name + "_thresholded"
        threshold = self._threshold_slider.value
        if self._invert_checkbox.value:
            thresholded = image < threshold
        else:
            thresholded = image > threshold
        if name in self._viewer.layers:
            self._viewer.layers[name].data = thresholded
        else:
            self._viewer.add_labels(thresholded, name=name)


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


class RLWorker(QObject):

    def __init__(self):
        super().__init__()
        self._piplines = {}


class RLDwidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.events.inserted.connect(self._on_change_layer)
        self.viewer.layers.events.removed.connect(self._on_change_layer)
        # self.viewer.layers.events.changed.connect(self._on_change_layer)

        self.thread = (
            QThread()
        )  # To prevent freezing GUIs, https://realpython.com/python-pyqt-qthread/
        self._widgets = {}
        # self._worker  =

        self.setLayout(QGridLayout())

        # RAW data box
        self.layout().addWidget(QLabel("Input RAW data"), 0, 0)
        self.input_raw_data_box = QComboBox()
        self.layout().addWidget(self.input_raw_data_box, 0, 1)

        # method
        self.layout().addWidget(QLabel("Method"), 1, 0)
        self.method_box = QComboBox()
        self.layout().addWidget(self.method_box, 1, 1)
        self.method_box.currentTextChanged.connect(self._on_change_method)

        # method parameters
        parameters_widget = QWidget()
        self.parameters_layout = QVBoxLayout()
        self.parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_widget.setLayout(self.parameters_layout)
        self.layout().addWidget(parameters_widget, 2, 0, 1, 2)

        self.run_btn = QPushButton("run")
        self.run_btn.clicked.connect(self._on_click)
        self.layout().addWidget(self.run_btn, 3, 0, 1, 2)

        # init the view
        self._on_change_layer()
        # self._on_change_method(self.method_box.currentText())

    def _on_click(self):
        print("run")

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
        print("method change.")


if __name__ == "__main__":
    viewer = napari.Viewer()
    dock, widget = viewer.window.add_plugin_dock_widget(
        "napari-kld", "RL Deconvolution"
    )

    napari.run()
