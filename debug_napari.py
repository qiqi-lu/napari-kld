# for debugging during development

from napari import Viewer, run

viewer = Viewer()
dw, widget = viewer.window.add_plugin_dock_widget(
    "napari-kld", "RL Deconvolution"
)

run()
