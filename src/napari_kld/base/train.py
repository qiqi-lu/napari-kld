import os
import time

import numpy as np
import skimage.io as io
import torch
from fft_conv_pytorch import fft_conv
from napari_kld.base.models import kernelnet
from napari_kld.base.utils import dataset_utils
from napari_kld.base.utils import evaluation as eva
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(
    dataset_name,
    data_path,
    output_path,
    psf_path=None,
    data_dim=3,
    num_iter=2,
    ks_z=1,
    ks_xy=31,
    model_name="kernet_fp",
):
    torch.manual_seed(7)

    device, num_workers = torch.device("cpu"), 0
    # device, num_workers = torch.device("cpu"), 6
    # device, num_workers = torch.device("cuda"), 6

    checkpoint_path = os.path.join(output_path, "checkpoints")
    data_range = None
    scale_factor = 1

    if data_dim == 2:
        kernel_size_fp = (ks_xy,) * 2
        kernel_size_bp = (ks_xy,) * 2

    if data_dim == 3:
        kernel_size_fp = [ks_z, ks_xy, ks_xy]
        kernel_size_bp = [ks_z, ks_xy, ks_xy]

    id_range = [0, 1]
    training_data_size = id_range[1] - id_range[0]
    batch_size = training_data_size
    epochs = 10000

    # --------------------------------------------------------------------------
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
    # model_name = 'kernet_fp'
    model_name = "kernet"
    # --------------------------------------------------------------------------
    if model_name == "kernet_fp":
        model_suffix = f"_ks_{ks_z}_{ks_xy}"
        multi_out = False
        self_supervised = False
        loss_main = torch.nn.MSELoss()

        optimizer_type = "adam"
        start_learning_rate = 0.001

        # optimizer_type = 'lbfgs'
        # start_learning_rate = 1

        epochs = 500

    if model_name == "kernet":
        num_iter = 2
        lam = 0.0  # lambda for prior
        multi_out = False
        shared_bp = True
        self_supervised = False
        # self_supervised = True

        ss_marker = "_ss" if self_supervised else ""
        model_suffix = f"_iter_{num_iter}_ks_{ks_z}_{ks_xy}{ss_marker}"
        loss_main = torch.nn.MSELoss()
        optimizer_type = "adam"

        # start_learning_rate = 0.000001 if self_supervised else 0.00001
        start_learning_rate = 0.000001 if self_supervised else 0.000001

        epochs = 10000
        # start_learning_rate = 0.000001
        # epochs = 7500

    # --------------------------------------------------------------------------
    warm_up = 0
    use_lr_schedule = True
    scheduler_cus = {}
    scheduler_cus["lr"] = start_learning_rate
    scheduler_cus["every"] = 2000  # 300
    scheduler_cus["rate"] = 0.5
    scheduler_cus["min"] = 0.00000001

    # --------------------------------------------------------------------------
    if data_dim == 2:
        if model_name == "kernet":
            save_every_iter, plot_every_iter = 1000, 50
            print_every_iter = 1000
        if model_name == "kernet_fp":
            save_every_iter, plot_every_iter = 5, 2
            print_every_iter = 1000

    if data_dim == 3:
        if model_name == "kernet":
            save_every_iter, plot_every_iter = 1000, 50
            print_every_iter = 1000
        if model_name == "kernet_fp":
            save_every_iter, plot_every_iter = 5, 2
            print_every_iter = 1000

        # --------------------------------------------------------------------------
        # Data
        # --------------------------------------------------------------------------
        # Training data
        hr_data_path = os.path.join(data_path, "gt")
        lr_data_path = os.path.join(data_path, "raw")
        hr_txt_file_path = os.path.join(data_path, "train.txt")
        lr_txt_file_path = hr_txt_file_path
        normalization, in_channels = (False, False), 1

    print(">> Load datasets from:", lr_data_path)

    # --------------------------------------------------------------------------
    # Training data
    training_data = dataset_utils.SRDataset(
        hr_data_path=hr_data_path,
        lr_data_path=lr_data_path,
        hr_txt_file_path=hr_txt_file_path,
        lr_txt_file_path=lr_txt_file_path,
        normalization=normalization,
        id_range=id_range,
    )

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
        # FP_type, BP_type = 'pre-trained', None
        # FP_type, BP_type = "known", None
        FP_type = "known"
        # --------------------------------------------------------------------------
        if FP_type == "pre-trained":
            print(">> Pred-trained PSF")

            # load FP parameters
            FP = kernelnet.ForwardProject(
                dim=data_dim,
                in_channels=in_channels,
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

            FP_path = os.path.join(
                "checkpoints",
                dataset_name,
                f"kernet_fp_bs_{batch_size}_lr_0.01_sf_{scale_factor}_mse_over2_inter_norm",
                "epoch_10000.pt",
            )  # 10000 (NF), 5000 (N)

            FP_para = torch.load(FP_path, map_location=device)
            FP.load_state_dict(FP_para["model_state_dict"])
            FP.eval()

            print(">> Load from: ", FP_path)

        if FP_type == "known":
            print(">> Known PSF")
            if data_dim == 2:
                ks, std = 25, 2.0
                ker = kernelnet.gauss_kernel_2d(shape=[ks, ks], std=std).to(
                    device=device
                )
                ker = ker.repeat(repeats=(in_channels, 1, 1, 1))

                def padd_fp(x):
                    return torch.nn.functional.pad(
                        input=x,
                        pad=(ks // 2, ks // 2, ks // 2, ks // 2),
                        mode=padding_mode,
                    )

                def conv_fp(x):
                    return torch.nn.functional.conv2d(
                        input=padd_fp(x), weight=ker, groups=in_channels
                    )

                def FP(x):
                    return torch.nn.functional.avg_pool2d(
                        conv_fp(x), kernel_size=25, stride=scale_factor
                    )

            if data_dim == 3:
                PSF_true = io.imread(psf_path).astype(np.float32)
                PSF_true = torch.tensor(PSF_true[None, None]).to(device=device)
                PSF_true = torch.round(PSF_true, decimals=16)
                ks = PSF_true.shape

                def padd_fp(x):
                    return torch.nn.functional.pad(
                        input=x,
                        pad=(
                            ks[-1] // 2,
                            ks[-1] // 2,
                            ks[-2] // 2,
                            ks[-2] // 2,
                            ks[-3] // 2,
                            ks[-3] // 2,
                        ),
                        mode=padding_mode,
                    )

                if conv_mode == "direct":

                    def conv_fp(x):
                        return torch.nn.functional.conv3d(
                            input=padd_fp(x),
                            weight=PSF_true,
                            groups=in_channels,
                        )

                if conv_mode == "fft":

                    def conv_fp(x):
                        return fft_conv(
                            signal=padd_fp(x),
                            kernel=PSF_true,
                            groups=in_channels,
                        )

                def FP(x):
                    return torch.nn.functional.avg_pool3d(
                        conv_fp(x),
                        kernel_size=scale_factor,
                        stride=scale_factor,
                    )

                print(">> Load from :", psf_path)

        # --------------------------------------------------------------------------
        model = kernelnet.KernelNet(
            dim=data_dim,
            in_channels=in_channels,
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
        model = kernelnet.ForwardProject(
            dim=data_dim,
            in_channels=in_channels,
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
    eva.count_parameters(model)
    print(model)

    # --------------------------------------------------------------------------
    # save
    if model_name == "kernet_fp":
        path_model = os.path.join(
            checkpoint_path,
            dataset_name,
            "forward",
            f"{model_name}_bs_{batch_size}_lr_{start_learning_rate}{model_suffix}",
        )

    if model_name == "kernet":
        path_model = os.path.join(
            checkpoint_path,
            dataset_name,
            "backward",
            f"{model_name}_bs_{batch_size}_lr_{start_learning_rate}{model_suffix}",
        )

    writer = SummaryWriter(os.path.join(path_model, "log"))
    print(">> Save model to", path_model)

    # --------------------------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------------------------
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=start_learning_rate
        )
    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=start_learning_rate,
            line_search_fn="strong_wolfe",
        )

    print(">> Start training ... ")
    print(time.asctime(time.localtime(time.time())))

    num_batches = len(train_dataloader)

    print(f">> Number of training batches: {num_batches}")

    if self_supervised:
        print("Training under self-supervised mode.")

    if training_data_size == 1:
        sample = training_data[0]
        x, y = sample["lr"].to(device)[None], sample["hr"].to(device)[None]
        y = y
    else:
        x, y = [], []
        for i in range(training_data_size):
            sample = training_data[i]
            x.append(sample["lr"])
            y.append(sample["hr"])
        x = torch.stack(x)
        y = torch.stack(y)
        x, y = x.to(device), y.to(device)
        y = y

    for i_epoch in range(epochs):
        print("\n" + "-" * 98)
        print(
            f"Epoch: {i_epoch + 1}/{epochs} | Batch size: {batch_size} | Num of Batches: {num_batches}"
        )
        print("-" * 98)
        # --------------------------------------------------------------------------
        ave_ssim, ave_psnr = 0, 0
        print_loss, print_ssim, print_psnr = [], [], []

        start_time = time.time()
        # --------------------------------------------------------------------------
        model.train()
        # for i_batch, sample in enumerate(train_dataloader):
        for i_batch in range(num_batches):
            i_iter = i_batch + i_epoch * num_batches  # index of iteration
            # ----------------------------------------------------------------------
            # load data
            # x, y = sample['lr'].to(device), sample['hr'].to(device)
            # y = y * ratio

            if model_name == "kernet_fp":
                inpt, gt = y, x
            if model_name == "kernet":
                if self_supervised:
                    inpt, gt = x, x
                else:
                    inpt, gt = x, y

            # ----------------------------------------------------------------------
            # optimize
            if optimizer_type == "lbfgs":
                # L-BFGS
                loss = 0.0
                pred = 0.0

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

            # ----------------------------------------------------------------------
            # custom learning rate scheduler
            if use_lr_schedule:
                if (warm_up > 0) and (i_iter < warm_up):
                    lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
                    # set learning rate
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

            # ----------------------------------------------------------------------
            if not multi_out:
                out = pred
            if multi_out:
                out = pred[-1]

            # ----------------------------------------------------------------------
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

            # ----------------------------------------------------------------------
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
            print("#", end="")

            if i_iter % print_every_iter == 0:
                print(
                    f"\nEpoch: {i_epoch}, Iter: {i_iter}, loss: {np.mean(print_loss):>.5f}, PSNR: {np.mean(print_psnr):>.5f},\
                    SSIM: {np.mean(print_ssim):>.5f}"
                )
                print(f"Computation time: {time.time()-start_time:>.2f} s")
                start_time = time.time()
                print_loss, print_ssim, print_psnr = [], [], []

            # ----------------------------------------------------------------------
            # save model and relative information
            if i_iter % save_every_iter == 0:
                print(
                    f"\nSave model ... (Epoch: {i_epoch}, Iteration: {i_iter})"
                )
                model_dict = {"model_state_dict": model.state_dict()}
                torch.save(
                    model_dict, os.path.join(path_model, f"epoch_{i_iter}.pt")
                )

    # --------------------------------------------------------------------------
    # save the last one model
    print(f"\nSave model ... (Epoch: {i_epoch}, Iteration: {i_iter+1})")
    model_dict = {"model_state_dict": model.state_dict()}
    torch.save(model_dict, os.path.join(path_model, f"epoch_{i_iter + 1}.pt"))

    # --------------------------------------------------------------------------
    writer.flush()
    writer.close()
    print("Training done!")
