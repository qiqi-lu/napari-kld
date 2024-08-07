# the widgets for each methods
import os

import napari
import numpy as np
import qtpy.QtCore
import skimage.io as io
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_kld.base import methods, predict, train
from napari_kld.base.generate_synthetic_data import generate_simulation_data
from napari_kld.base.utils.dataset_utils import get_image_shape, read_txt
from napari_kld.baseww import (
    DirectorySelectWidget,
    DoubleSpinBox,
    FileSelectWidget,
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
        self.iteration_box = SpinBox(vmin=1, vmax=1000, vinit=30)
        self.iteration_box.valueChanged.connect(self._on_num_iter_change)
        self.layout_grid.addWidget(self.iteration_box, 0, 1)
        self.progress_bar.setMaximum(30)

        self.params_group.setLayout(self.layout_grid)
        self.layout.addWidget(self.params_group)

        self.layout.setAlignment(qtpy.QtCore.Qt.AlignTop)

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
        self.image_name = ""

    def set_image(self, image):
        self.image_name = image.name
        self.image = image.data

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
        if not isinstance(self._out, int):
            self.viewer.add_image(
                self._out,
                name=f"{self.image_name}_deconv_{self.widget.label.lower()}_iter_{self.widget.iteration_box.value()}",
            )


# RLD using Guassian kernel
class WidgetRLDeconvGaussian(WidgetRLDeconvTraditional):
    def __init__(self, progress_bar):
        super().__init__(progress_bar)
        self.label = "Gaussian"
        self.params_group.setTitle(self.label)
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
        self.params_group.setTitle(self.label)

        self.iteration_box.setValue(30)
        self.progress_bar.setMaximum(30)

        # beta box
        self.layout_grid.addWidget(QLabel("beta:"), 1, 0)
        self.beta_box = DoubleSpinBox(vmin=0, vmax=10, vinit=0.01)
        self.beta_box.setDecimals(5)
        self.beta_box.setSingleStep(0.01)
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
        self.params_group.setTitle(self.label)

        # alpha box
        self.layout_grid.addWidget(QLabel("alpha"), 3, 0)
        self.alpha_box = DoubleSpinBox(vmin=0, vmax=10, vinit=0.005)
        self.alpha_box.setDecimals(5)
        self.alpha_box.setSingleStep(0.001)
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
        self.logger = logger
        self.img_shape = None

        self.setTitle("Model Training")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # ----------------------------------------------------------------------
        self.fp_widget = WidgetKLDeconvTrainFP(logger=logger)
        grid_layout.addWidget(self.fp_widget, 4, 0, 1, 3)

        # ----------------------------------------------------------------------
        self.bp_widget = WidgetKLDeconvTrainBP(logger=logger)
        grid_layout.addWidget(self.bp_widget, 5, 0, 1, 3)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Data Directory"), 0, 0, 1, 1)
        self.data_directory_box = DirectorySelectWidget(
            title="Select the directory of training data"
        )
        self.data_directory_box.path_edit.textChanged.connect(
            self._on_path_change
        )
        self.data_directory_box.path_edit.textChanged.connect(
            self._on_params_change
        )
        grid_layout.addWidget(self.data_directory_box, 0, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Output Directory"), 1, 0, 1, 1)
        self.output_directory_box = DirectorySelectWidget(
            title="Select the directory of output"
        )
        self.output_directory_box.path_edit.textChanged.connect(
            self._on_path_change
        )
        self.output_directory_box.path_edit.textChanged.connect(
            self._on_params_change
        )
        grid_layout.addWidget(self.output_directory_box, 1, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("PSF Directory"), 2, 0, 1, 1)
        self.psf_path_box = FileSelectWidget(title="Select the PSF")
        self.psf_path_box.path_edit.textChanged.connect(
            self._on_psf_path_change
        )
        self.psf_path_box.path_edit.textChanged.connect(self._on_params_change)
        grid_layout.addWidget(self.psf_path_box, 2, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Dimension"), 3, 0, 1, 1)
        self.dim_box = QLineEdit()
        self.dim_box.setText("2")
        self.dim_box.setReadOnly(True)
        self.dim_box.textChanged.connect(self._on_params_change)
        grid_layout.addWidget(self.dim_box, 3, 1, 1, 1)

        self.preprocess_check_box = QCheckBox()
        self.preprocess_check_box.setText("Preprocess")
        grid_layout.addWidget(self.preprocess_check_box, 3, 2, 1, 1)

        grid_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)

        # ----------------------------------------------------------------------
        # init
        self.enable_run(False)
        self._on_params_change()

    def _on_psf_path_change(self):
        psf_path = self.psf_path_box.get_path()

        if psf_path != "":
            self.fp_widget.setVisible(False)
            if os.path.exists(psf_path):
                self.bp_widget.fp_path_box.set_enabled(False)
            else:
                show_info(f'ERROR: "{psf_path}" does not exist.')
                self.enable_run(False)
        else:
            self.fp_widget.setVisible(True)

    def _on_path_change(self):
        path_data = self.data_directory_box.get_path()
        path_output = self.output_directory_box.get_path()

        # check path exist
        if path_data != "" and not os.path.exists(path_data):
            show_info(f'ERROR: "{path_data}" does not exist.')

        if path_output != "" and not os.path.exists(path_output):
            show_info(f'ERROR: "{path_output}" does not exist.')

        # enable run
        if (
            path_data != ""
            and path_output != ""
            and os.path.exists(path_data)
            and os.path.exists(path_output)
        ):
            self.check_training_data()
            if self.img_shape is not None:
                dim = len(self.img_shape)
                self.dim_box.setText(str(dim))
                if dim == 2:
                    self.fp_widget.ks_box_z.setValue(1)
                    self.bp_widget.ks_box_z.setValue(1)
                if dim == 3:
                    self.fp_widget.ks_box_z.setValue(3)
                    self.bp_widget.ks_box_z.setValue(3)
        else:
            self.enable_run(False)

    def enable_run(self, enable: bool):
        self.fp_widget.enable_run(enable)
        self.bp_widget.enable_run(enable)

    def _on_params_change(self):
        dim = int(self.dim_box.text())
        data_path = self.data_directory_box.get_path()
        output_path = self.output_directory_box.get_path()
        psf_path = self.psf_path_box.get_path()
        preprocess = 1 if self.preprocess_check_box.checkState() else 0

        params_dict = {
            "data_dim": dim,
            "num_channel": 1,
            "data_path": data_path,
            "output_path": output_path,
            "psf_path": psf_path,
            "preprocess": preprocess,
        }

        self.fp_widget.set_params(params_dict)
        self.bp_widget.set_params(params_dict)

    def log(self, text):
        print(text)
        if self.logger is not None:
            self.logger.add_text(text)

    def check_training_data(self):
        path = self.data_directory_box.get_path()

        if os.path.exists(os.path.join(path, "train.txt")):
            if os.path.exists(os.path.join(path, "raw")):
                if not os.path.exists(os.path.join(path, "gt")):
                    show_info("WARNNING: the [gt] folder does not exist.")

                name_list = read_txt(os.path.join(path, "train.txt"))
                if len(name_list) > 0:
                    img_path = os.path.join(path, "raw", name_list[0])
                    if os.path.exists(img_path):
                        img = io.imread(img_path)
                        self.img_shape = img.shape
                        self.log(f"Input image shape = {self.img_shape}")
                        self.enable_run(True)
                    else:
                        show_info("ERROR : Image does not exist.")
                        self.enable_run(False)
                else:
                    show_info(
                        "ERROR : No image name is listed in train.txt file."
                    )
                    self.enable_run(False)
            else:
                show_info("ERROR: the [raw] folder does not exist.")
                self.enable_run(False)
        else:
            show_info("ERROR: the [train.txt] file does not exist.")
            self.enable_run(False)


class WorkerKLDeconvTrainFP(WorkerBase):
    def __init__(self, observer):
        super().__init__(observer=observer)
        self.abort_flag = [False]

    def run(self):
        self.log("-" * 80)
        self.log("Start training Forward Projection ...")
        self.abort_flag = [False]
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
            self.log("Run failed.")
        self.finish_signal.emit()
        self.log("-" * 80)

    def stop(self):
        self.abort_flag[0] = True
        self.finish_signal.emit()


class WidgetKLDeconvTrainFP(WidgetBase):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self._worker = WorkerKLDeconvTrainFP(self._observer)

        self.setTitle("Forward Projection")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Epoch | Batch Size"), 1, 0, 1, 1)
        self.epoch_box = SpinBox(vmin=1, vmax=20000, vinit=100)
        self.bs_box = SpinBox(vmin=1, vmax=1000, vinit=1)
        grid_layout.addWidget(self.epoch_box, 1, 1, 1, 1)
        grid_layout.addWidget(self.bs_box, 1, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Kernel Size (z, xy)"), 2, 0, 1, 1)
        self.ks_box_z = SpinBox(vmin=1, vmax=999, vinit=1)
        self.ks_box_z.setSingleStep(2)
        self.ks_box_xy = SpinBox(vmin=3, vmax=999, vinit=31)
        self.ks_box_xy.setSingleStep(2)
        self.ks_box_z.valueChanged.connect(self._on_ks_change)
        self.ks_box_xy.valueChanged.connect(self._on_ks_change)
        grid_layout.addWidget(self.ks_box_z, 2, 1, 1, 1)
        grid_layout.addWidget(self.ks_box_xy, 2, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Optimizer | Learning Rate"), 3, 0, 1, 1)
        self.optim_box = QComboBox()
        self.optim_box.addItems(["Adam", "LBFGS"])
        self.lr_box = DoubleSpinBox(vmin=0, vmax=10, vinit=0.001, decimals=7)
        self.lr_box.setSingleStep(0.001)
        grid_layout.addWidget(self.optim_box, 3, 1, 1, 1)
        grid_layout.addWidget(self.lr_box, 3, 2, 1, 1)

        grid_layout.addWidget(QLabel("Decay Step | Decay Rate"), 4, 0, 1, 1)
        self.decay_step_box = SpinBox(vmin=0, vmax=100000, vinit=0)
        self.decay_step_box.setSingleStep(1000)
        self.decay_rate_box = DoubleSpinBox(
            vmin=0, vmax=1, vinit=0, decimals=2
        )
        self.decay_rate_box.setSingleStep(0.1)
        grid_layout.addWidget(self.decay_step_box, 4, 1, 1, 1)
        grid_layout.addWidget(self.decay_rate_box, 4, 2, 1, 1)

        # ----------------------------------------------------------------------
        self.stop_btn = QPushButton("stop")
        grid_layout.addWidget(self.run_btn, 5, 0, 1, 2)
        grid_layout.addWidget(self.stop_btn, 5, 2, 1, 1)
        grid_layout.addWidget(self.progress_bar, 6, 0, 1, 3)

        # ----------------------------------------------------------------------
        # init
        self.enable_run(False)

        # ----------------------------------------------------------------------
        # connect
        self.stop_btn.clicked.connect(self._on_click_stop)
        self.reconnect()

    def _on_click_run(self):
        self.restart()

        params_dict = self.get_params()
        self.print_params(params_dict)
        self.progress_bar.setMaximum(params_dict["num_epoch"])

        self._worker.set_params(params_dict)
        self._thread.start()

    def _on_click_stop(self):
        self._worker.stop()

    def set_params(self, params_dict):
        self._worker.set_params(params_dict)

    def get_params(self):
        ks_z = self.ks_box_z.value()
        ks_xy = self.ks_box_xy.value()
        num_epoch = self.epoch_box.value()
        batch_size = self.bs_box.value()
        learning_rate = self.lr_box.value()
        optimizer = self.optim_box.currentText()
        decay_step = self.decay_step_box.value()
        decay_rate = self.decay_rate_box.value()

        params_dict = {
            "num_epoch": num_epoch,
            "batch_size": batch_size,
            "ks_z": ks_z,
            "ks_xy": ks_xy,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "decay_step": decay_step,
            "decay_rate": decay_rate,
        }

        return params_dict

    def _on_ks_change(self):
        ks_z = self.ks_box_z.value()
        if (ks_z % 2) == 0:
            ks_z += 1
            self.ks_box_z.setValue(ks_z)

        ks_xy = self.ks_box_xy.value()
        if (ks_xy % 2) == 0:
            ks_xy += 1
            self.ks_box_xy.setValue(ks_xy)


class WorkerKLDeconvTrainBP(WorkerBase):
    def __init__(self, observer):
        super().__init__(observer=observer)
        self.abort_flag = [False]

    def run(self):
        self.log("-" * 80)
        self.log("start training Backward Projection ...")
        self.abort_flag = [False]
        try:
            train.train(
                model_name="kernet",
                observer=self.observer,
                abort_flag=self.abort_flag,
                **self.params_dict,
            )
        except (RuntimeError, TypeError) as e:
            print(str(e))
            self.log("Run failed.")
        self.finish_signal.emit()
        self.log("-" * 80)

    def stop(self):
        self.abort_flag[0] = True
        self.finish_signal.emit()


class WidgetKLDeconvTrainBP(WidgetBase):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self._worker = WorkerKLDeconvTrainBP(self._observer)

        self.setTitle("Backward Projection")
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Training Strategy"), 0, 0, 1, 1)
        self.training_strategy_box = QComboBox()
        self.training_strategy_box.addItems(["supervised", "self-supervised"])
        grid_layout.addWidget(self.training_strategy_box, 0, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Iterations (RL)"), 1, 0, 1, 1)
        self.iteration_box_rl = SpinBox(vmin=1, vmax=99, vinit=2)
        grid_layout.addWidget(self.iteration_box_rl, 1, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Epoch | Batch Size"), 2, 0, 1, 1)
        self.epoch_box = SpinBox(vmin=1, vmax=20000, vinit=1000)
        self.bs_box = SpinBox(vmin=1, vmax=1000, vinit=1)
        grid_layout.addWidget(self.epoch_box, 2, 1, 1, 1)
        grid_layout.addWidget(self.bs_box, 2, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Kernel Size (z, xy)"), 3, 0, 1, 1)
        self.ks_box_z = SpinBox(vmin=1, vmax=1000, vinit=1)
        self.ks_box_z.setSingleStep(2)
        self.ks_box_xy = SpinBox(vmin=3, vmax=999, vinit=31)
        self.ks_box_xy.setSingleStep(2)
        self.ks_box_z.valueChanged.connect(self._on_ks_change)
        self.ks_box_xy.valueChanged.connect(self._on_ks_change)
        grid_layout.addWidget(self.ks_box_z, 3, 1, 1, 1)
        grid_layout.addWidget(self.ks_box_xy, 3, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("FP Directory"), 4, 0, 1, 1)
        self.fp_path_box = FileSelectWidget("Select Forward Projection model")
        grid_layout.addWidget(self.fp_path_box, 4, 1, 1, 2)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Optimizer | Learning Rate"), 5, 0, 1, 1)
        self.optim_box = QComboBox()
        self.optim_box.addItems(["Adam", "LBFGS"])
        self.learning_rate_box = DoubleSpinBox(
            vmin=0, vmax=10, vinit=0.000001, decimals=9
        )
        self.learning_rate_box.setSingleStep(0.00001)
        grid_layout.addWidget(self.optim_box, 5, 1, 1, 1)
        grid_layout.addWidget(self.learning_rate_box, 5, 2, 1, 1)

        grid_layout.addWidget(QLabel("Decay Step | Decay Rate"), 6, 0, 1, 1)
        self.decay_step_box = SpinBox(vmin=0, vmax=100000, vinit=2000)
        self.decay_step_box.setSingleStep(1000)
        self.decay_rate_box = DoubleSpinBox(
            vmin=0, vmax=1, vinit=0.5, decimals=2
        )
        self.decay_rate_box.setSingleStep(0.1)
        grid_layout.addWidget(self.decay_step_box, 6, 1, 1, 1)
        grid_layout.addWidget(self.decay_rate_box, 6, 2, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(self.run_btn, 7, 0, 1, 2)
        self.stop_btn = QPushButton("stop")
        grid_layout.addWidget(self.stop_btn, 7, 2, 1, 1)
        grid_layout.addWidget(self.progress_bar, 8, 0, 1, 3)

        # ----------------------------------------------------------------------
        # init
        self.enable_run(False)

        # ----------------------------------------------------------------------
        self.stop_btn.clicked.connect(self._on_click_stop)
        self.reconnect()

    def _on_click_run(self):
        self.restart()

        params_dict = self.get_params()
        self.print_params(params_dict)
        self.progress_bar.setMaximum(params_dict["num_epoch"])

        self._worker.set_params(params_dict)
        self._thread.start()

    def _on_click_stop(self):
        print("stop")
        self._worker.stop()

    def set_params(self, params_dict):
        self._worker.set_params(params_dict=params_dict)

    def get_params(self):
        fp_path = self.fp_path_box.get_path()
        num_iter = self.iteration_box_rl.value()
        ks_z = self.ks_box_z.value()
        ks_xy = self.ks_box_xy.value()
        num_epoch = self.epoch_box.value()
        batch_size = self.bs_box.value()
        training_strategy = self.training_strategy_box.currentText()

        if training_strategy == "self-supervised":
            self_supervised = True
        else:
            self_supervised = False

        learning_rate = self.learning_rate_box.value()
        decay_step = self.decay_step_box.value()
        decay_rate = self.decay_rate_box.value()
        optimizer = self.optim_box.currentText()

        params_dict = {
            "fp_path": fp_path,
            "num_iter": num_iter,
            "ks_z": ks_z,
            "ks_xy": ks_xy,
            "num_epoch": num_epoch,
            "batch_size": batch_size,
            "self_supervised": self_supervised,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "decay_step": decay_step,
            "decay_rate": decay_rate,
        }

        return params_dict

    def _on_ks_change(self):
        ks_z = self.ks_box_z.value()
        if (ks_z % 2) == 0:
            ks_z += 1
            self.ks_box_z.setValue(ks_z)

        ks_xy = self.ks_box_xy.value()
        if (ks_xy % 2) == 0:
            ks_xy += 1
            self.ks_box_xy.setValue(ks_xy)


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
        output_name = f"{self.input_name}_deconv_iter_{num_iter}"
        self.log(f'Save as "{output_name}"')

        self.viewer.add_image(
            self.img_output,
            name=output_name,
        )

    def set_input(self, name):
        self.input_name = name

    def run(self):
        img_input = self.viewer.layers[self.input_name].data

        self.log("-" * 80)
        self.log("start predicting ...")
        self.log(f"input shape: {img_input.shape}")

        try:
            self.img_output = predict.predict(
                img_input, observer=self.observer, **self.params_dict
            )

            if not isinstance(self.img_output, int):
                show_info("Succeed")
                self.log(f"output shape: {self.img_output.shape}")
                self.succeed_signal.emit()
            else:
                show_info("Failed")

        except (RuntimeError, TypeError) as e:
            self.log(str(e))
            show_info("Failed")

        self.finish_signal.emit()
        self.log("-" * 80)


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
        self.psf_path_box.path_edit.textChanged.connect(
            self._on_psf_path_change
        )
        grid_layout.addWidget(self.psf_path_box, 1, 1, 1, 2)

        # ----------------------------------------------------------------------
        # forward projection model
        grid_layout.addWidget(QLabel("Forward Projection"), 2, 0, 1, 1)
        self.fp_path_box = FileSelectWidget()
        self.fp_path_box.path_edit.textChanged.connect(self._on_fp_path_change)
        grid_layout.addWidget(self.fp_path_box, 2, 1, 1, 2)

        # ----------------------------------------------------------------------
        # backward projeciton model
        grid_layout.addWidget(QLabel("Backward Projection"), 3, 0, 1, 1)
        self.bp_path_box = FileSelectWidget()
        self.bp_path_box.path_edit.textChanged.connect(self._on_bp_path_change)
        grid_layout.addWidget(self.bp_path_box, 3, 1, 1, 2)

        # ----------------------------------------------------------------------
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
        self.restart()
        self.progress_bar.setMaximum(self.iteration_box_rl.value())

        params_dict = self.get_params()
        self.print_params(params_dict)

        self._worker.set_params(params_dict)
        self._worker.set_input(self.input_raw_data_box.currentText())
        self._thread.start()

    def _on_change_layer(self):
        self.input_raw_data_box.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.input_raw_data_box.addItem(layer.name)

        if self.input_raw_data_box.count() < 1:
            self.enable_run(False)

    def _on_bp_path_change(self):
        bp_path = self.bp_path_box.get_path()
        psf_path = self.psf_path_box.get_path()
        fp_path = self.fp_path_box.get_path()

        if bp_path != "":
            if os.path.exists(bp_path):
                _, ext = os.path.splitext(bp_path)
                if ext == ".pt":
                    self.log(f'Backward Projeciton Directory: "{bp_path}"')
                    if psf_path != "" or fp_path != "":
                        self.enable_run(True)
                else:
                    self.enable_run(False)
                    show_info(
                        'ERROR: the backward projection should be a model with a ".pt" extension.'
                    )
            else:
                show_info(f'ERROR: "{bp_path}" does not exist.')
                self.enable_run(False)
        else:
            self.enable_run(False)

    def _on_fp_path_change(self):
        fp_path = self.fp_path_box.get_path()

        if fp_path != "":
            if os.path.exists(fp_path):
                _, ext = os.path.splitext(fp_path)
                if ext == ".pt":
                    self.log(f'Forward Projeciton Directory: "{fp_path}"')
                else:
                    show_info(
                        'ERROR: the forward projection should be a model with a ".pt" extension.'
                    )
                    self.enable_run(False)
            else:
                show_info(f'ERROR: "{fp_path}" does not exist.')
                self.enable_run(False)
        else:
            psf_path = self.psf_path_box.get_path()
            if psf_path == "":
                show_info("ERROR : a PSF or forward projeciton is required.")
                self.enable_run(False)

    def _on_psf_path_change(self):
        psf_path = self.psf_path_box.get_path()

        if psf_path != "":
            if os.path.exists(psf_path):
                _, ext = os.path.splitext(psf_path)
                if ext == ".tif":
                    self.log(f'PSF Directory: "{psf_path}"')
                else:
                    show_info(
                        'ERROR: the PSF file should be with a ".tif" extension.'
                    )
                    self.enable_run(False)
            else:
                show_info(f'ERROR: "{psf_path}" does not exist.')
                self.enable_run(False)
        else:
            fp_path = self.fp_path_box.get_path()
            if fp_path == "":
                show_info("ERROR : a PSF or forward projeciton is required.")
                self.enable_run(False)

    def _on_succeed(self):
        self._worker.set_output()


class WorkerKLDeconvSimulation(WorkerBase):
    def __init__(self, observer):
        super().__init__(observer)

    def run(self):
        self.log("simulation data generating ...")
        try:
            generate_simulation_data(
                observer=self.observer, **self.params_dict
            )
        except (RuntimeError, TypeError, UnboundLocalError) as e:
            self.log(str(e))
            self.log("Run failed.")

        self.finish_signal.emit()
        self.log("finished.")


class WidgetKLDeconvSimulation(WidgetBase):
    def __init__(self, logger=None):
        super().__init__(logger)
        self._worker = WorkerKLDeconvSimulation(self._observer)
        self.psf_shape = None

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
        grid_layout.addWidget(QLabel("PSF Directory"), 1, 0, 1, 1)
        self.psf_path_box = FileSelectWidget()
        self.psf_path_box.path_edit.textChanged.connect(self._on_change_path)
        grid_layout.addWidget(self.psf_path_box, 1, 1, 1, 3)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Image Shape (z,x,y)"), 2, 0, 1, 1)
        self.shape_z_box = SpinBox(vmin=1, vmax=999, vinit=128)
        self.shape_y_box = SpinBox(vmin=32, vmax=999, vinit=128)
        self.shape_x_box = SpinBox(vmin=32, vmax=999, vinit=128)
        self.shape_z_box.valueChanged.connect(self.set_psf_range)
        self.shape_y_box.valueChanged.connect(self.set_psf_range)
        self.shape_x_box.valueChanged.connect(self.set_psf_range)
        grid_layout.addWidget(self.shape_z_box, 2, 1, 1, 1)
        grid_layout.addWidget(self.shape_y_box, 2, 2, 1, 1)
        grid_layout.addWidget(self.shape_x_box, 2, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("PSF Crop (z,x,y)"), 3, 0, 1, 1)
        self.crop_z_box = SpinBox(vmin=1, vmax=999, vinit=0)
        self.crop_y_box = SpinBox(vmin=3, vmax=999, vinit=0)
        self.crop_x_box = SpinBox(vmin=3, vmax=999, vinit=0)
        self.crop_z_box.setSingleStep(2)
        self.crop_y_box.setSingleStep(2)
        self.crop_x_box.setSingleStep(2)
        self.crop_z_box.valueChanged.connect(self._on_psf_crop_shape_change)
        self.crop_y_box.valueChanged.connect(self._on_psf_crop_shape_change)
        self.crop_x_box.valueChanged.connect(self._on_psf_crop_shape_change)
        grid_layout.addWidget(self.crop_z_box, 3, 1, 1, 1)
        grid_layout.addWidget(self.crop_y_box, 3, 2, 1, 1)
        grid_layout.addWidget(self.crop_x_box, 3, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(QLabel("Num of Simulations"), 4, 0, 1, 1)
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

        grid_layout.addWidget(QLabel("Scale Factor"), 6, 2, 1, 1)
        self.sf_box = SpinBox(vmin=1, vmax=10, vinit=1)
        grid_layout.addWidget(self.sf_box, 6, 3, 1, 1)

        # ----------------------------------------------------------------------
        grid_layout.addWidget(self.run_btn, 7, 0, 1, 4)
        grid_layout.addWidget(self.progress_bar, 8, 0, 1, 4)

        # ----------------------------------------------------------------------
        self._on_change_path()
        self.reconnect()

    def set_psf_range(self):
        dim_psf = len(self.psf_shape)
        dim_data = 2 if self.shape_z_box.value() == 1 else 3

        if dim_data == dim_psf:
            if dim_psf == 3:
                ks_z, ks_y, ks_x = self.psf_shape

                if (ks_z % 2) == 0:
                    ks_z -= 1
                if (ks_y % 2) == 0:
                    ks_y -= 1
                if (ks_x % 2) == 0:
                    ks_x -= 1

                self.crop_z_box.setMaximum(
                    np.minimum(ks_z, self.shape_z_box.value())
                )
                self.crop_z_box.setMinimum(3)
                self.crop_y_box.setMaximum(
                    np.minimum(ks_y, self.shape_y_box.value())
                )
                self.crop_x_box.setMaximum(
                    np.minimum(ks_x, self.shape_x_box.value())
                )

                self.crop_z_box.setValue(ks_z)
                self.crop_y_box.setValue(ks_y)
                self.crop_x_box.setValue(ks_x)

            if dim_psf == 2:
                ks_y, ks_x = self.psf_shape

                if (ks_y % 2) == 0:
                    ks_y -= 1
                if (ks_x % 2) == 0:
                    ks_x -= 1

                self.crop_z_box.setMaximum(1)
                self.crop_y_box.setMaximum(
                    np.minimum(ks_y, self.shape_y_box.value())
                )
                self.crop_x_box.setMaximum(
                    np.minimum(ks_x, self.shape_x_box.value())
                )

                self.crop_z_box.setValue(1)
                self.crop_y_box.setValue(ks_y)
                self.crop_x_box.setValue(ks_x)

            self.enable_run(True)
        else:
            show_info(f"ERROR : a {dim_data}D PSF is required.")
            self.enable_run(False)

    def get_params(self):
        data_path = self.output_path_box.get_path()
        psf_path = self.psf_path_box.get_path()
        image_shape = (
            int(self.shape_z_box.value()),
            int(self.shape_y_box.value()),
            int(self.shape_x_box.value()),
        )
        psf_crop_shape = (
            int(self.crop_z_box.value()),
            int(self.crop_y_box.value()),
            int(self.crop_x_box.value()),
        )

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
            "psf_crop_shape": psf_crop_shape,
            "std_gauss": std_gauss,
            "poisson": poisson,
            "ratio": ratio,
            "scale_factor": scale_factor,
        }

        return params_dict

    def _on_click_run(self):
        self.restart()
        self.progress_bar.setMaximum(self.num_simu_box.value())
        params_dict = self.get_params()
        self.print_params(params_dict=params_dict)
        self._worker.set_params(params_dict=params_dict)
        self._thread.start()

    def _on_change_path(self):
        output_path = self.output_path_box.get_path()
        psf_path = self.psf_path_box.get_path()

        if (
            output_path != ""
            and psf_path != ""
            and os.path.exists(output_path)
        ):
            if os.path.exists(psf_path):
                _, ext = os.path.splitext(psf_path)

                if ext != ".tif":
                    show_info('ERROR : only support ".tif" file.')
                    self.enable_run(False)
                else:
                    self.psf_shape = get_image_shape(psf_path)
                    self.set_psf_range()
            else:
                show_info(f'ERROR: "{psf_path}" does not exsits.')
                self.enable_run(False)
        else:
            self.enable_run(False)

    def _on_psf_crop_shape_change(self):
        ks_z = self.crop_z_box.value()
        if (ks_z % 2) == 0:
            ks_z -= 1
            self.crop_z_box.setValue(ks_z)

        ks_y = self.crop_y_box.value()
        if (ks_y % 2) == 0:
            ks_y -= 1
            self.crop_y_box.setValue(ks_y)

        ks_x = self.crop_x_box.value()
        if (ks_x % 2) == 0:
            ks_x -= 1
            self.crop_x_box.setValue(ks_x)
