# the widgets for each methods
import os

import napari
import qtpy.QtCore
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_kld.base import methods, predict, train
from napari_kld.base.generate_synthetic_data import generate_simulation_data
from napari_kld.baseww import (
    DirectorySelectWidget,
    DoubleSpinBox,
    FileSelectWidget,
    ProgressObserver,
    SpinBox,
    WidgetBase,
    WorkerBase,
)


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


class WidgetKLDeconvTrain(QGroupBox):
    def __init__(self, logger=None):
        super().__init__()
        self.setTitle("Train")

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        self.fp_widget = WidgetKLDeconvTrainFP(logger=logger)
        grid_layout.addWidget(self.fp_widget, 4, 0, 1, 3)

        self.bp_widget = WidgetKLDeconvTrainBP(logger=logger)
        grid_layout.addWidget(self.bp_widget, 5, 0, 1, 3)

        grid_layout.addWidget(QLabel("Data Directory"), 0, 0, 1, 1)
        self.data_directory_widget = DirectorySelectWidget()
        self.data_directory_widget.path_edit.textChanged.connect(
            self._on_data_path_change
        )
        grid_layout.addWidget(self.data_directory_widget, 0, 1, 1, 2)

        grid_layout.addWidget(QLabel("Output Directory"), 1, 0, 1, 1)
        self.output_directory_widget = DirectorySelectWidget()
        self.output_directory_widget.path_edit.textChanged.connect(
            self._on_output_path_change
        )
        grid_layout.addWidget(self.output_directory_widget, 1, 1, 1, 2)

        grid_layout.addWidget(QLabel("PSF Directory"), 2, 0, 1, 1)
        self.psf_directory_widget = FileSelectWidget()
        self.psf_directory_widget.path_edit.textChanged.connect(
            self._on_psf_path_change
        )
        grid_layout.addWidget(self.psf_directory_widget, 2, 1, 1, 2)

        grid_layout.addWidget(QLabel("Image Channels/Dimension"), 3, 0, 1, 1)
        self.channel_box = SpinBox(vmin=1, vmax=10, vinit=1)
        self.channel_box.valueChanged.connect(self._on_channel_change)
        grid_layout.addWidget(self.channel_box, 3, 1, 1, 1)
        self.dim_box = QComboBox()
        self.dim_box.currentTextChanged.connect(self._on_dim_change)
        self.dim_box.addItems(["3", "2"])
        grid_layout.addWidget(self.dim_box, 3, 2, 1, 1)

        grid_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)

        # initialization
        self._on_dim_change()

    def _on_psf_path_change(self):
        print("psf path change")
        psf_path = self.psf_directory_widget.get_path()
        self.fp_widget.update_params_dict({"psf_path": psf_path})
        self.bp_widget.update_params_dict({"psf_path": psf_path})

        if psf_path != "":
            self.fp_widget.setVisible(False)
            if not os.path.exists(psf_path):
                napari.utils.notifications.show_info("ERROR: PSF Not Exists.")
                self.bp_widget.run_btn.setEnabled(False)
        else:
            self.fp_widget.setVisible(True)

    def _on_output_path_change(self):
        print("output path change")
        path = self.output_directory_widget.get_path()
        self.fp_widget.update_params_dict({"output_path": path})
        self.bp_widget.update_params_dict({"output_path": path})

    def _on_data_path_change(self):
        print("data path change")
        path = self.data_directory_widget.get_path()

        self.fp_widget.update_params_dict({"data_path": path})
        self.bp_widget.update_params_dict({"data_path": path})

        if path != "":
            # check path exist
            if os.path.exists(path):
                self.fp_widget.enable_run(True)
                self.bp_widget.enable_run(True)
            else:
                napari.utils.notifications.show_info(
                    "ERROR: Data Path Unexists."
                )
        else:
            self.fp_widget.enable_run(False)
            self.bp_widget.enable_run(False)

    def _on_dim_change(self):
        dim = self.dim_box.currentText()
        print(f"data dimension change to {int(dim)}")
        self.fp_widget.update_params_dict({"data_dim": int(dim)})
        self.bp_widget.update_params_dict({"data_dim": int(dim)})

    def _on_channel_change(self):
        num_channel = self.channel_box.value()
        print(f"data channel change to {num_channel}")
        self.fp_widget.update_params_dict({"num_channel": num_channel})
        self.bp_widget.update_params_dict({"num_channel": num_channel})


class WorkerKLDeconvTrainFP(QObject):
    finish_signal = Signal()

    def __init__(self, observer):
        super().__init__()
        self.observer = observer
        self.params_dict = {}
        self.abort_flag = [False]

    def run(self):
        print("worker run")
        try:
            train.train(
                fp_path=None,
                num_iter=1,
                model_name="kernet_fp",
                self_supervised=False,
                observer=self.observer,
                abort_flag=self.abort_flag,
                **self.params_dict,
            )
        except (RuntimeError, TypeError) as e:
            print(str(e))
            self.observer.notify("Run Failed.")
        self.finish_signal.emit()

    def stop(self):
        self.abort_flag[0] = True
        self.finish_signal.emit()

    def set_params(self, params_dict):
        self.params_dict = params_dict


class WidgetKLDeconvTrainFP(QGroupBox):
    def __init__(self, logger=None):
        super().__init__()
        self.params_dict = {
            "data_path": "",
            "output_path": "",
            "psf_path": "",
            "data_dim": 3,
            "num_channel": 1,
            "num_epoch": 100,
            "batch_size": 1,
            "ks_z": 1,
            "ks_xy": 31,
            "learning_rate": 0.001,
        }

        self._observer = ProgressObserver()
        self._worker = WorkerKLDeconvTrainFP(self._observer)
        self._thread = QThread()
        self.logger = logger

        self.setTitle("Forward Projection")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Epoch/Batch Size"), 1, 0, 1, 1)
        self.epoch_box = SpinBox(vmin=1, vmax=10000, vinit=100)
        self.epoch_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.epoch_box, 1, 1, 1, 1)

        self.bs_box = SpinBox(vmin=1, vmax=1000, vinit=1)
        self.bs_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.bs_box, 1, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Kernel Size (z, xy)"), 2, 0, 1, 1)
        self.ks_box_z = SpinBox(vmin=1, vmax=1000, vinit=1)
        self.ks_box_z.setSingleStep(2)
        self.ks_box_z.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.ks_box_z, 2, 1, 1, 1)

        self.ks_box_xy = SpinBox(vmin=3, vmax=999, vinit=31)
        self.ks_box_xy.setSingleStep(2)
        self.ks_box_xy.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.ks_box_xy, 2, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Learning rate"), 3, 0, 1, 1)
        self.lr_box = QDoubleSpinBox()
        self.lr_box.setSingleStep(0.001)
        self.lr_box.setMinimum(0)
        self.lr_box.setDecimals(7)
        self.lr_box.setValue(0.001)
        self.lr_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.lr_box, 3, 1, 1, 2)

        # ----------------------------------------------------------------------
        self.run_btn = QPushButton("run")
        grid_layout.addWidget(self.run_btn, 4, 0, 1, 2)
        self.run_btn.clicked.connect(self._on_click_run)

        self.stop_btn = QPushButton("stop")
        grid_layout.addWidget(self.stop_btn, 4, 2, 1, 1)
        self.stop_btn.clicked.connect(self._on_click_stop)

        # ----------------------------------------------------------------------
        self.progress_bar = QProgressBar()
        grid_layout.addWidget(self.progress_bar, 5, 0, 1, 3)

        # ----------------------------------------------------------------------
        # init
        self.enable_run(False)
        self._on_param_change()

        # ----------------------------------------------------------------------
        # connect
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finish_signal.connect(self._thread.quit)

        self._observer.progress_signal.connect(self._on_progress)
        self._observer.notify_signal.connect(self._on_notify)

    def _on_click_run(self):
        print("run")
        self._on_notify("Parameters: ")
        for item in self.params_dict:
            self._on_notify(f"{item} : {self.params_dict[item]}")
        self._worker.set_params(self.params_dict)
        self._thread.start()

    def _on_click_stop(self):
        self._worker.stop()

    def enable_run(self, enable):
        self.run_btn.setEnabled(enable)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_notify(self, value):
        if self.logger is not None:
            self.logger.add_text(value)

    def update_params_dict(self, path_dict):
        self.params_dict.update(path_dict)

    def _on_param_change(self):
        ks_z = self.ks_box_z.value()
        if (ks_z % 2) == 0:
            ks_z += 1
            self.ks_box_z.setValue(ks_z)

        ks_xy = self.ks_box_xy.value()
        if (ks_xy % 2) == 0:
            ks_xy += 1
            self.ks_box_xy.setValue(ks_xy)

        num_epoch = self.epoch_box.value()
        batch_size = self.bs_box.value()
        learning_rate = self.lr_box.value()

        self.params_dict.update(
            {
                "ks_z": ks_z,
                "ks_xy": ks_xy,
                "num_epoch": num_epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )


class WorkerKLDeconvTrainBP(QObject):
    finish_signal = Signal()

    def __init__(self, observer):
        super().__init__()
        self.observer = observer
        self.params_dict = {}
        self.abort_flag = [False]

    def run(self):
        print("start training ...")
        try:
            train.train(
                model_name="kernet",
                observer=self.observer,
                abort_flag=self.abort_flag,
                **self.params_dict,
            )
            print("training done.")
        except (RuntimeError, TypeError) as e:
            print(str(e))
            self.observer.notify("Run Failed.")
        self.finish_signal.emit()

    def set_params(self, params_dict):
        self.params_dict = params_dict

    def stop(self):
        self.abort_flag[0] = True
        self.finish_signal.emit()


class WidgetKLDeconvTrainBP(QGroupBox):
    def __init__(self, logger=None):
        super().__init__()
        self.params_dict = {
            "data_path": "",
            "output_path": "",
            "psf_path": "",
            "fp_path": "",
            "num_channel": 1,
            "data_dim": 3,
            "num_iter": 2,
            "ks_z": 1,
            "ks_xy": 31,
            "num_epoch": 10000,
            "batch_size": 1,
            "self_supervised": False,
            "learning_rate": 0.000001,
        }

        self._observer = ProgressObserver()
        self._worker = WorkerKLDeconvTrainBP(self._observer)
        self._thread = QThread()
        self.logger = logger

        self.setTitle("Backward Projection")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Training strategy"), 0, 0, 1, 1)
        self.training_strategy_box = QComboBox()
        self.training_strategy_box.addItems(
            ["supervised training", "self-supervised training"]
        )
        self.training_strategy_box.currentTextChanged.connect(
            self._on_param_change
        )
        grid_layout.addWidget(self.training_strategy_box, 0, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Iterations (RL)"), 1, 0, 1, 1)
        self.iteration_box_rl = SpinBox(vmin=1, vmax=99, vinit=2)
        self.iteration_box_rl.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.iteration_box_rl, 1, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Epoch/Batch Size"), 2, 0, 1, 1)
        self.epoch_box = SpinBox(vmin=1, vmax=10000, vinit=100)
        self.epoch_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.epoch_box, 2, 1, 1, 1)

        self.bs_box = SpinBox(vmin=1, vmax=1000, vinit=1)
        self.bs_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.bs_box, 2, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Kernel Size (z, xy)"), 3, 0, 1, 1)
        self.ks_box_z = SpinBox(vmin=1, vmax=1000, vinit=1)
        self.ks_box_z.setSingleStep(2)
        self.ks_box_z.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.ks_box_z, 3, 1, 1, 1)

        self.ks_box_xy = SpinBox(vmin=3, vmax=999, vinit=31)
        self.ks_box_xy.setSingleStep(2)
        self.ks_box_xy.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.ks_box_xy, 3, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("FP directory"), 4, 0, 1, 1)
        self.fp_path_box = FileSelectWidget()
        self.fp_path_box.path_edit.textChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.fp_path_box, 4, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Learning Rate"), 5, 0, 1, 1)
        self.learning_rate_box = QDoubleSpinBox()
        self.learning_rate_box.setMinimum(0)
        self.learning_rate_box.setSingleStep(0.000001)
        self.learning_rate_box.setDecimals(9)
        self.learning_rate_box.setValue(0.000001)
        self.learning_rate_box.valueChanged.connect(self._on_param_change)
        grid_layout.addWidget(self.learning_rate_box, 5, 1, 1, 2)

        # ----------------------------------------------------------------------
        self.run_btn = QPushButton("run")
        grid_layout.addWidget(self.run_btn, 6, 0, 1, 2)
        self.run_btn.clicked.connect(self._on_click_run)

        self.stop_btn = QPushButton("stop")
        grid_layout.addWidget(self.stop_btn, 6, 2, 1, 1)
        self.stop_btn.clicked.connect(self._on_click_stop)

        self.progress_bar = QProgressBar()
        grid_layout.addWidget(self.progress_bar, 7, 0, 1, 3)

        # ----------------------------------------------------------------------
        # init
        self.enable_run(False)
        self._on_param_change()

        # ----------------------------------------------------------------------
        # connect
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finish_signal.connect(self._thread.quit)
        self._observer.progress_signal.connect(self._on_progress)
        self._observer.notify_signal.connect(self._on_notify)

    def _on_click_run(self):
        print("[BP run]")
        self._on_notify("Parameters: ")
        for item in self.params_dict:
            self._on_notify(f"{item} : {self.params_dict[item]}")
        self._thread.quit()
        self._worker.set_params(self.params_dict)
        self._thread.start()

    def _on_click_stop(self):
        print("user stops")
        self._worker.stop()

    def enable_run(self, enable):
        self.run_btn.setEnabled(enable)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_notify(self, value):
        if self.logger is not None:
            self.logger.add_text(value)

    def update_params_dict(self, path_dict):
        self.params_dict.update(path_dict)

    def _on_param_change(self):
        self.progress_bar.setValue(0)
        num_iter = self.iteration_box_rl.value()
        num_epoch = self.epoch_box.value()
        batch_size = self.bs_box.value()
        learning_rate = self.learning_rate_box.value()
        fp_path = self.fp_path_box.get_path()

        ks_z = self.ks_box_z.value()
        if (ks_z % 2) == 0:
            ks_z += 1
            self.ks_box_z.setValue(ks_z)

        ks_xy = self.ks_box_xy.value()
        if (ks_xy % 2) == 0:
            ks_xy += 1
            self.ks_box_xy.setValue(ks_xy)

        training_strategy = self.training_strategy_box.currentText()
        if training_strategy == "self-supervised learning":
            self_supervised = True
        else:
            self_supervised = False

        self.params_dict.update(
            {
                "fp_path": fp_path,
                "num_iter": num_iter,
                "ks_z": ks_z,
                "ks_xy": ks_xy,
                "num_epoch": num_epoch,
                "batch_size": batch_size,
                "self_supervised": self_supervised,
                "learning_rate": learning_rate,
            }
        )


class WorkerKLDeconvPredict(WorkerBase):
    finish_signal = Signal()
    succeed_signal = Signal()

    def __init__(self, viewer, observer):
        super().__init__(observer=observer)
        self.viewer = viewer
        self.input_name = ""
        self.img_output = None

    def set_output(self):
        num_iter = self.params_dict["num_iter"]

        self.viewer.add_image(
            self.img_output,
            name=f"{self.input_name}_deconv_iter_{num_iter}",
        )

    def set_input(self, name):
        self.input_name = name

    def run(self):
        img_input = self.viewer.layers[self.input_name].data

        self.observer.notify("start predicting ...")
        self.observer.notify(f"input shape: {img_input.shape}")

        try:
            self.img_output = predict.predict(
                img_input, observer=self.observer, **self.params_dict
            )
            if not isinstance(self.img_output, int):
                self.observer.pop_info("Succeed")
                self.observer.notify(f"output shape: {self.img_output.shape}")
                self.succeed_signal.emit()
            else:
                self.observer.pop_info("Failed")

        except (RuntimeError, TypeError) as e:
            self.observer.notify(str(e))
            self.observer.pop_info("Failed")

        self.finish_signal.emit()


class WidgetKLDeconvPredict(WidgetBase):
    def __init__(self, viewer: napari.Viewer, logger=None):
        super().__init__(logger=logger)
        self.viewer = viewer
        self.viewer.layers.events.inserted.connect(self._on_change_layer)
        self.viewer.layers.events.removed.connect(self._on_change_layer)

        self._worker = WorkerKLDeconvPredict(self.viewer, self._observer)

        self.setTitle("Predict")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        # RAW data box
        grid_layout.addWidget(QLabel("Input RAW data"), 0, 0, 1, 1)
        self.input_raw_data_box = QComboBox()
        grid_layout.addWidget(self.input_raw_data_box, 0, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("PSF directory"), 1, 0, 1, 1)
        self.psf_path_box = FileSelectWidget()
        grid_layout.addWidget(self.psf_path_box, 1, 1, 1, 2)

        # ----------------------------------------------------------------------
        # forward projection model
        grid_layout.addWidget(QLabel("Forward Projection"), 2, 0, 1, 1)
        self.fp_path_box = FileSelectWidget()
        grid_layout.addWidget(self.fp_path_box, 2, 1, 1, 2)

        # backward projeciton model
        grid_layout.addWidget(QLabel("Backward Projection"), 3, 0, 1, 1)
        self.bp_path_box = FileSelectWidget()
        self.bp_path_box.path_edit.textChanged.connect(self._on_bp_path_change)
        grid_layout.addWidget(self.bp_path_box, 3, 1, 1, 2)

        grid_layout.addWidget(QLabel("Iterations (RL)"), 4, 0, 1, 1)
        self.iteration_box_rl = SpinBox(vmin=1, vmax=999, vinit=2)
        grid_layout.addWidget(self.iteration_box_rl, 4, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(self.run_btn, 5, 0, 1, 3)
        grid_layout.addWidget(self.progress_bar, 6, 0, 1, 3)

        # ----------------------------------------------------------------------
        # initialization
        self._on_change_layer()
        self._on_bp_path_change()

        # ----------------------------------------------------------------------
        # connect the thread
        self.reconnect()
        self._worker.succeed_signal.connect(self._on_succeed)

    def get_params(self):
        psf_path = self.psf_path_box.get_path()
        fp_path = self.fp_path_box.get_path()
        bp_path = self.bp_path_box.get_path()
        num_iter = self.iteration_box_rl.value()

        params_dict = {
            "psf_path": psf_path,
            "fp_path": fp_path,
            "bp_path": bp_path,
            "num_iter": num_iter,
        }
        return params_dict

    def _on_click_run(self):
        print("Predicting ...")
        self.progress_bar.setValue(0)
        self._thread.quit()

        params_dict = self.get_params()
        self._on_notify("Parameters:")
        for item in params_dict:
            self._on_notify(f"{item} : {params_dict[item]}")

        self._worker.set_params(params_dict)
        self._worker.set_input(self.input_raw_data_box.currentText())
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

    def _on_bp_path_change(self):
        bp_path = self.bp_path_box.get_path()

        if bp_path != "":
            if not os.path.exists(bp_path):
                self.enable_run(False)
            else:
                self.enable_run(True)
        else:
            self.enable_run(False)

    def _on_succeed(self):
        self._worker.set_output()


class WorkerKLDeconvSimulation(WorkerBase):
    def __init__(self, observer):
        super().__init__(observer)

    def run(self):
        print("run simulation worker ...")
        try:
            generate_simulation_data(
                observer=self.observer, **self.params_dict
            )
        except (RuntimeError, TypeError, UnboundLocalError) as e:
            print(str(e))
            self.observer.notify("Run failed.")
        self.finish_signal.emit()
        print("finished.")


class WidgetKLDeconvSimulation(WidgetBase):
    def __init__(self, logger=None):
        super().__init__(logger)
        self._worker = WorkerKLDeconvSimulation(self._observer)

        self.setTitle("Simulation")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Output Directory"), 0, 0, 1, 1)
        self.output_path_box = DirectorySelectWidget()
        self.output_path_box.path_edit.textChanged.connect(
            self._on_change_path
        )
        grid_layout.addWidget(self.output_path_box, 0, 1, 1, 3)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("PSF directory"), 1, 0, 1, 1)
        self.psf_path_box = FileSelectWidget()
        self.psf_path_box.path_edit.textChanged.connect(self._on_change_path)
        grid_layout.addWidget(self.psf_path_box, 1, 1, 1, 3)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Image shape (z,x,y)"), 2, 0, 1, 1)
        self.shape_z_box = SpinBox(vmin=1, vmax=999, vinit=128)
        self.shape_y_box = SpinBox(vmin=32, vmax=999, vinit=128)
        self.shape_x_box = SpinBox(vmin=32, vmax=999, vinit=128)
        grid_layout.addWidget(self.shape_z_box, 2, 1, 1, 1)
        grid_layout.addWidget(self.shape_y_box, 2, 2, 1, 1)
        grid_layout.addWidget(self.shape_x_box, 2, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("PSF crop (z,x,y)"), 3, 0, 1, 1)
        self.crop_z_box = SpinBox(vmin=0, vmax=999, vinit=0)
        grid_layout.addWidget(self.crop_z_box, 3, 1, 1, 1)
        self.crop_y_box = SpinBox(vmin=0, vmax=999, vinit=0)
        grid_layout.addWidget(self.crop_y_box, 3, 2, 1, 1)
        self.crop_x_box = SpinBox(vmin=0, vmax=999, vinit=0)
        grid_layout.addWidget(self.crop_x_box, 3, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Num of simulation"), 4, 0, 1, 1)
        self.num_simu_box = SpinBox(vmin=1, vmax=999, vinit=1)
        grid_layout.addWidget(self.num_simu_box, 4, 1, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Gaussian (std)"), 5, 0, 1, 1)
        self.gauss_std_box = DoubleSpinBox(vmin=0, vmax=999, vinit=0)
        grid_layout.addWidget(self.gauss_std_box, 5, 1, 1, 1)

        grid_layout.addWidget(QLabel("Poisson"), 5, 2, 1, 1)
        self.poiss_check_box = QCheckBox()
        self.poiss_check_box.setText("Enable")
        grid_layout.addWidget(self.poiss_check_box, 5, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Ratio"), 6, 0, 1, 1)
        self.ratio_box = DoubleSpinBox(vmin=0, vmax=99, vinit=1)
        grid_layout.addWidget(self.ratio_box, 6, 1, 1, 1)

        grid_layout.addWidget(QLabel("Scale factor"), 6, 2, 1, 1)
        self.sf_box = SpinBox(vmin=1, vmax=10, vinit=1)
        grid_layout.addWidget(self.sf_box, 6, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(self.run_btn, 7, 0, 1, 4)
        grid_layout.addWidget(self.progress_bar, 8, 0, 1, 4)

        # ----------------------------------------------------------------------
        self._on_change_path()
        self.reconnect()

    def get_params(self):
        data_path = self.output_path_box.get_path()
        psf_path = self.psf_path_box.get_path()
        image_shape = (
            int(self.shape_z_box.value()),
            int(self.shape_y_box.value()),
            int(self.shape_x_box.value()),
        )
        psf_crop = (
            int(self.crop_z_box.value()),
            int(self.crop_y_box.value()),
            int(self.crop_x_box.value()),
        )

        if psf_crop[0] == 0 or psf_crop[1] == 0 or psf_crop[2] == 0:
            psf_crop = None

        num_simulation = self.num_simu_box.value()
        std_gauss = self.gauss_std_box.value()
        poisson = 1 if self.poiss_check_box.checkState() else 0
        ratio = self.ratio_box.value()
        scale_factor = self.sf_box.value()

        params_dict = {
            "path_dataset": data_path,
            "path_psf": psf_path,
            "image_shape": image_shape,
            "num_simulation": num_simulation,
            "psf_crop_shape": psf_crop,
            "std_gauss": std_gauss,
            "poisson": poisson,
            "ratio": ratio,
            "scale_factor": scale_factor,
        }

        return params_dict

    def _on_click_run(self):
        print("Simulation data generating ...")
        self.restart()
        self.progress_bar.setMaximum(self.num_simu_box.value())
        params_dict = self.get_params()
        self.print_params(params_dict=params_dict)
        self._worker.set_params(params_dict=params_dict)
        self._thread.start()

    def _on_change_path(self):
        data_path = self.output_path_box.get_path()
        psf_path = self.psf_path_box.get_path()

        if (
            data_path == ""
            or psf_path == ""
            or (not os.path.exists(data_path) or not os.path.exists(psf_path))
        ):
            self.enable_run(False)
        else:
            self.enable_run(True)
