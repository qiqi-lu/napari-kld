import json
import os
import pathlib
import time

import numpy as np
import skimage.io as io
import torch
from napari.utils.notifications import show_info
from napari_kld.base.models import kernelnet
from napari_kld.base.utils import dataset_utils
from napari_kld.base.utils import evaluation as eva

# from fft_conv_pytorch import fft_conv
from napari_kld.base.utils.dataset_utils import fftn_conv as fft_conv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(
    data_path,
    output_path,
    psf_path=None,
    fp_path=None,
    num_channel=1,
    data_dim=3,
    num_iter=2,
    ks_z=31,
    ks_xy=31,
    model_name="kernet_fp",  # "kernet" or "kernet_fp"
    num_epoch=10000,
    batch_size=1,
    self_supervised=False,
    learning_rate=0.001,  # start learning rate
    optimizer="Adam",
    decay_step=0,
    decay_rate=0,
    observer=None,
    abort_flag=None,
    preprocess=0,
    **kwargs,
):

    def notify(value):
        print(value)
        if observer is not None:
            observer.notify(value)

    # custom parameters
    torch.manual_seed(7)

    if psf_path != "":
        FP_type = "known"
    elif fp_path != "":
        FP_type = "pre-trained"

    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    if self_supervised and psf_path == "":
        self_supervised = False
        show_info(
            "ERROR : can not use self-supervised training, a PSF is required."
        )

    # --------------------------------------------------------------------------
    # kernel size setting
    if data_dim == 2:
        kernel_size_fp = (ks_xy,) * 2
        kernel_size_bp = (ks_xy,) * 2

    if data_dim == 3:
        kernel_size_fp = [ks_z, ks_xy, ks_xy]
        kernel_size_bp = [ks_z, ks_xy, ks_xy]

    # --------------------------------------------------------------------------
    scale_factor = 1
    data_range = None
    id_range = None

    device, num_workers = torch.device("cpu"), 0

    # if torch.cuda.is_available():
    #     device, num_workers = torch.device("cuda"), 6
    # else:
    #     device, num_workers = torch.device("cpu"), 0

    conv_mode, padding_mode, kernel_init = "fft", "reflect", "gauss"
    interpolation = True
    kernel_norm_fp = False
    kernel_norm_bp = True
    over_sampling = 2

    if data_dim == 2:
        std_init = [2.0, 2.0]
    if data_dim == 3:
        std_init = [4.0, 2.0, 2.0]

    # --------------------------------------------------------------------------
    # forward projection
    if model_name == "kernet_fp":
        model_suffix = f"_ks_{ks_z}_{ks_xy}"
        multi_out, self_supervised = False, False
        loss_main = torch.nn.MSELoss()

    # backward projection
    if model_name == "kernet":
        lam, multi_out, shared_bp = 0.0, False, True
        ss_marker = "_ss" if self_supervised else ""
        model_suffix = f"_iter_{num_iter}_ks_{ks_z}_{ks_xy}{ss_marker}"
        loss_main = torch.nn.MSELoss()

        notify(f"Use {FP_type} forward projection.")

    # --------------------------------------------------------------------------
    optimizer_type, start_learning_rate = str(optimizer), learning_rate
    # optimizer_type, start_learning_rate = 'LBFGS', 1

    warm_up = 0
    scheduler_cus = {}
    scheduler_cus["lr"] = start_learning_rate

    if decay_step > 0:
        notify(
            f"use learning rate scheduler, start learning rate = {start_learning_rate}, decay step = {decay_step}, decay_rate = {decay_rate}"
        )
        use_lr_schedule = True
        scheduler_cus["every"] = decay_step  # 300
        scheduler_cus["rate"] = decay_rate
        scheduler_cus["min"] = 0.00000001
    else:
        notify(f"use fixed learning rate = {learning_rate}")
        use_lr_schedule = False

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------
    # Training data
    data_path = pathlib.Path(data_path)
    notify(f"load training data set from: {data_path}")

    hr_data_path = pathlib.Path(data_path, "gt")
    lr_data_path = pathlib.Path(data_path, "raw")
    if self_supervised:
        hr_data_path = lr_data_path
    hr_txt_file_path = pathlib.Path(data_path, "train.txt")
    lr_txt_file_path = hr_txt_file_path

    if not os.path.exists(hr_txt_file_path):
        notify("ERROR: Training data does not exists.")
        return 0

    # --------------------------------------------------------------------------
    # Training data
    if preprocess == 1:
        notify("Enable image preprocessing.")

    training_data = dataset_utils.SRDataset(
        hr_root_path=hr_data_path,
        lr_root_path=lr_data_path,
        hr_txt_file_path=hr_txt_file_path,
        lr_txt_file_path=lr_txt_file_path,
        normalization=(False, False),
        id_range=id_range,
        preprocess=preprocess,
    )

    training_data_size = training_data.__len__()

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    if model_name == "kernet":
        FP, BP = None, None

        if FP_type == "pre-trained":
            print("use pred-trained forward projection ...")

            # check the parameters of pre-trained forward projection
            parent = pathlib.Path(fp_path).parent
            with open(pathlib.Path(parent, "parameters.json")) as f:
                params_fp = json.load(f)

            if params_fp["data_dim"] != data_dim:
                data_dim_pre = params_fp["data_dim"]
                print(
                    f"ERROR: training data dim of FP is {data_dim_pre}, current data dim is {data_dim}"
                )
                return 0
            if params_fp["num_channel"] != num_channel:
                num_channel_pre = params_fp["num_channel"]
                print(
                    f"ERROR: training data channel of FP is {num_channel_pre}, current data channel is {num_channel}"
                )
                return 0
            if params_fp["ks_z"] != ks_z or params_fp["ks_xy"] != ks_xy:
                ks_z_pre, ks_xy_pre = params_fp["ks_z"], params_fp["ks_xy"]
                print(
                    f"ERROR: kernel size of pre-trained FP is ({ks_z_pre}, {ks_xy_pre}), current kernel size is ({ks_z}, {ks_xy})"
                )
                return 0

            # create forward projection model
            FP = kernelnet.ForwardProject(
                dim=data_dim,
                in_channels=num_channel,
                scale_factor=scale_factor,
                kernel_size=kernel_size_fp,
                std_init=std_init,
                padding_mode=padding_mode,
                init=kernel_init,
                trainable=False,
                interpolation=interpolation,
                kernel_norm=kernel_norm_fp,
                over_sampling=over_sampling,
                conv_mode=conv_mode,
            )

            # load weights of froward projection model
            FP_para = torch.load(fp_path, map_location=device)
            FP.load_state_dict(FP_para["model_state_dict"])
            FP.eval()

            print("load forward projection model from: ", fp_path)
            if observer is not None:
                observer.notify(
                    f"load forward projection model from: {fp_path}"
                )

        if FP_type == "known":
            notify("known PSF")
            PSF_true = io.imread(psf_path).astype(np.float32)
            PSF_true = torch.tensor(PSF_true[None, None]).to(device=device)
            PSF_true = torch.round(PSF_true, decimals=16)
            ks = PSF_true.shape

            def padd_fp(x):
                if data_dim == 3:
                    pad_size = (
                        ks[-1] // 2,
                        ks[-1] // 2,
                        ks[-2] // 2,
                        ks[-2] // 2,
                        ks[-3] // 2,
                        ks[-3] // 2,
                    )
                if data_dim == 2:
                    pad_size = (
                        ks[-1] // 2,
                        ks[-1] // 2,
                        ks[-2] // 2,
                        ks[-2] // 2,
                    )
                x_pad = torch.nn.functional.pad(
                    input=x,
                    pad=pad_size,
                    mode=padding_mode,
                )
                return x_pad

            if conv_mode == "direct":

                def conv_fp(x):
                    if data_dim == 3:
                        x_conv = torch.nn.functional.conv3d(
                            input=padd_fp(x),
                            weight=PSF_true,
                            groups=num_channel,
                        )
                    if data_dim == 2:
                        x_conv = torch.nn.functional.conv2d(
                            input=padd_fp(x),
                            weight=PSF_true,
                            groups=num_channel,
                        )
                    return x_conv

            if conv_mode == "fft":

                def conv_fp(x):
                    return fft_conv(
                        signal=padd_fp(x),
                        kernel=PSF_true,
                        groups=num_channel,
                    )

            def FP(x):
                if data_dim == 3:
                    x_fp = torch.nn.functional.avg_pool3d(
                        conv_fp(x),
                        kernel_size=scale_factor,
                        stride=scale_factor,
                    )
                if data_dim == 2:
                    x_fp = torch.nn.functional.avg_pool2d(
                        conv_fp(x),
                        kernel_size=scale_factor,
                        stride=scale_factor,
                    )
                return x_fp

            notify(f"load psf from {psf_path}")

        # ----------------------------------------------------------------------
        # create whole algorithm
        model = kernelnet.KernelNet(
            dim=data_dim,
            in_channels=num_channel,
            scale_factor=scale_factor,
            num_iter=num_iter,
            kernel_size_fp=kernel_size_fp,
            kernel_size_bp=kernel_size_bp,
            std_init=std_init,
            init=kernel_init,
            FP=FP,
            BP=BP,
            lam=lam,
            padding_mode=padding_mode,
            multi_out=multi_out,
            interpolation=interpolation,
            kernel_norm=kernel_norm_bp,
            over_sampling=over_sampling,
            shared_bp=shared_bp,
            self_supervised=self_supervised,
            conv_mode=conv_mode,
        ).to(device)

    # --------------------------------------------------------------------------
    if model_name == "kernet_fp":
        # create fprward projection model
        model = kernelnet.ForwardProject(
            dim=data_dim,
            in_channels=num_channel,
            scale_factor=scale_factor,
            kernel_size=kernel_size_fp,
            std_init=std_init,
            init=kernel_init,
            padding_mode=padding_mode,
            trainable=True,
            kernel_norm=kernel_norm_fp,
            interpolation=interpolation,
            conv_mode=conv_mode,
            over_sampling=over_sampling,
        ).to(device)

    # --------------------------------------------------------------------------
    info = eva.count_parameters(model)
    print(model)

    # save
    if model_name == "kernet_fp":
        path_model = os.path.join(
            checkpoint_path,
            f"forward_bs_{batch_size}_lr_{start_learning_rate}{model_suffix}",
        )

    if model_name == "kernet":
        path_model = os.path.join(
            checkpoint_path,
            f"backward_bs_{batch_size}_lr_{start_learning_rate}{model_suffix}",
        )

    writer = SummaryWriter(os.path.join(path_model, "log"))
    notify(info)
    notify(f"save model to {path_model}")

    # --------------------------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------------------------
    notify(f"use {optimizer_type} optimizer.")
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=start_learning_rate
        )
    if optimizer_type == "LBFGS":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=start_learning_rate,
            line_search_fn="strong_wolfe",
        )

    notify("start training ... ")
    notify(time.asctime(time.localtime(time.time())))

    if self_supervised:
        notify("Training under self-supervised mode.")

    # --------------------------------------------------------------------------
    # load data
    if batch_size == training_data_size:
        # load all data in one batch
        x, y = [], []
        for i in range(training_data_size):
            sample = training_data[i]
            x.append(sample["lr"])
            y.append(sample["hr"])
        x, y = torch.stack(x), torch.stack(y)
        x, y = x.to(device), y.to(device)
        num_batches = 1
    elif batch_size > training_data_size:
        notify(
            "ERROR: the batch size should not be larger than training data size."
        )
        return 0
    else:
        num_batches = len(train_dataloader)

    notify(f"number of training batches: {num_batches}")

    num_iter_training = num_epoch * num_batches
    save_every_iter = int(num_iter_training * 0.1)
    plot_every_iter = np.round(num_iter_training * 0.005)
    print_every_iter = int(num_iter_training * 0.1)

    # --------------------------------------------------------------------------
    notify(
        f"Epoch: {num_epoch} | Batch size: {batch_size} | Num of Batches: {num_batches}"
    )
    # --------------------------------------------------------------------------
    for i_epoch in range(num_epoch):
        if abort_flag is not None and abort_flag[0]:
            notify("Abort!")
            return 0

        if observer is not None:
            observer.progress(i_epoch + 1)
        ave_ssim, ave_psnr = 0, 0
        print_loss, print_ssim, print_psnr = [], [], []

        start_time = time.time()
        model.train()

        if batch_size < training_data_size:
            sample_iter = train_dataloader

        if batch_size == training_data_size:
            sample_iter = range(num_batches)

        for i_batch, sample in enumerate(sample_iter):
            i_iter = i_batch + i_epoch * num_batches  # index of iteration

            if batch_size < training_data_size:
                x = sample["lr"].to(device)
                y = sample["hr"].to(device)

            if model_name == "kernet_fp":
                inpt, gt = y, x

            if model_name == "kernet":
                if self_supervised:
                    inpt, gt = x, x
                else:
                    inpt, gt = x, y

            # ------------------------------------------------------------------
            # optimize
            if optimizer_type == "LBFGS":
                # L-BFGS optimization
                loss, pred = 0.0, 0.0

                def closure(x, y):
                    global loss
                    global pred
                    pred = model(x)
                    optimizer.zero_grad()
                    loss = loss_main(pred, y)
                    loss.backward()
                    return loss

                optimizer.step(closure(inpt, gt))

            else:
                optimizer.zero_grad()
                pred = model(inpt)
                loss = loss_main(pred, gt)
                loss.backward()
                optimizer.step()

            # ------------------------------------------------------------------
            # custom learning rate scheduler
            if use_lr_schedule:
                if (warm_up > 0) and (i_iter < warm_up):
                    lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                if (
                    i_iter >= warm_up
                    and (i_iter + 1 - warm_up) % scheduler_cus["every"] == 0
                ):
                    lr = scheduler_cus["lr"] * (
                        scheduler_cus["rate"]
                        ** ((i_iter + 1 - warm_up) // scheduler_cus["every"])
                    )
                    lr = np.maximum(lr, scheduler_cus["min"])
                    for g in optimizer.param_groups:
                        g["lr"] = lr
            else:
                if (warm_up > 0) and (i_iter < warm_up):
                    lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                if i_iter >= warm_up:
                    for g in optimizer.param_groups:
                        g["lr"] = scheduler_cus["lr"]

            # ------------------------------------------------------------------
            out = pred if not multi_out else pred[-1]

            # plot loss and metrics
            if i_iter % plot_every_iter == 0:
                if data_dim == 2:
                    ave_ssim, ave_psnr = eva.measure_2d(
                        img_test=out, img_true=gt, data_range=data_range
                    )
                if data_dim == 3:
                    ave_ssim, ave_psnr = eva.measure_3d(
                        img_test=out, img_true=gt, data_range=data_range
                    )
                if writer is not None:
                    writer.add_scalar("loss", loss, i_iter)
                    writer.add_scalar("psnr", ave_psnr, i_iter)
                    writer.add_scalar("ssim", ave_ssim, i_iter)
                    writer.add_scalar(
                        "Leanring Rate",
                        optimizer.param_groups[-1]["lr"],
                        i_iter,
                    )

            # ------------------------------------------------------------------
            # print and save model
            if data_dim == 2:
                s, p = eva.measure_2d(
                    img_test=out, img_true=gt, data_range=data_range
                )
            if data_dim == 3:
                s, p = eva.measure_3d(
                    img_test=out, img_true=gt, data_range=data_range
                )
            print_loss.append(loss.cpu().detach().numpy())
            print_ssim.append(s)
            print_psnr.append(p)

            if i_iter % print_every_iter == 0:
                notify(
                    f"Epoch: {i_epoch}, Iter: {i_iter}, loss: {np.mean(print_loss):>.5f}, PSNR: {np.mean(print_psnr):>.5f}, SSIM: {np.mean(print_ssim):>.5f}"
                )
                notify(f"Computation time: {time.time()-start_time:>.2f} s")
                start_time = time.time()
                print_loss, print_ssim, print_psnr = [], [], []

            # ------------------------------------------------------------------
            # save model and relative information
            if i_iter % save_every_iter == 0:
                notify(
                    f"save model ... (Epoch: {i_epoch}, Iteration: {i_iter})"
                )
                model_dict = {"model_state_dict": model.state_dict()}
                torch.save(
                    model_dict, os.path.join(path_model, f"epoch_{i_iter}.pt")
                )

    # --------------------------------------------------------------------------
    # save the last one model
    notify(f"Save model ... (Epoch: {i_epoch}, Iteration: {i_iter+1})")
    model_dict = {"model_state_dict": model.state_dict()}
    torch.save(
        model_dict, os.path.join(path_model, f"epoch_{i_iter + 1}_final.pt")
    )

    # save parameters
    parameters_dict = {
        "data_path": str(data_path),
        "psf_path": str(psf_path),
        "fp_path": str(fp_path),
        "num_channel": num_channel,
        "data_dim": data_dim,
        "num_iter": num_iter,
        "ks_z": ks_z,
        "ks_xy": ks_xy,
        "model_name": model_name,  # "kernet" or "kernet_fp"
        "num_epoch": num_epoch,
        "batch_size": batch_size,
        "self_supervised": self_supervised,
        "learning_rate": learning_rate,  # start learning rate
        "optimizer": optimizer_type,
        "decay_step": decay_step,
        "decay_rate": decay_rate,
        "preprocess": preprocess,
    }
    with open(os.path.join(path_model, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters_dict))

    # --------------------------------------------------------------------------
    writer.flush()
    writer.close()
    notify("Training done!")


if __name__ == "__main__":
    # forward projection training
    train(
        data_path="D:/GitHub/napari-kld/test/data/real/3D/train",
        output_path="D:/GitHub/napari-kld/test/data/real/3D",
        psf_path="",
        fp_path="",
        num_channel=1,
        data_dim=3,
        num_iter=2,
        ks_z=3,
        ks_xy=31,
        model_name="kernet_fp",  # "kernet" or "kernet_fp"
        num_epoch=100,
        batch_size=1,
        self_supervised=False,
        learning_rate=0.001,  # start learning rate
        observer=None,
    )

    # backward projection training
    # train(
    #     data_path="D:/GitHub/napari-kld/test/data/real/3D/train",
    #     output_path="D:/GitHub/napari-kld/test/data/real/3D",
    #     psf_path="",
    #     fp_path="D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\checkpoints\\forward_bs_1_lr_0.001_ks_1_31\\epoch_100.pt",
    #     num_channel=1,
    #     data_dim=2,
    #     num_iter=2,
    #     ks_z=1,
    #     ks_xy=31,
    #     model_name="kernet",  # "kernet" or "kernet_fp"
    #     num_epoch=100,
    #     batch_size=1,
    #     self_supervised=False,
    #     learning_rate=0.00001,  # start learning rate
    #     observer=None,
    # )
