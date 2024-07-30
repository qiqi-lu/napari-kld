import json
import pathlib
import time

import numpy as np
import skimage.io as io
import torch
from fft_conv_pytorch import fft_conv
from napari_kld.base.models import kernelnet


def predict(
    img,
    psf_path="",
    fp_path="",
    bp_path="",
    num_iter=2,
    observer=None,
    **kwargs,
):
    data_dim = len(img.shape)
    in_channels = 1

    def notify(value):
        print(value)
        if observer is not None:
            observer.notify(value)

    device = torch.device("cpu")

    if psf_path != "":
        notify("Use a known PSF as forward kernel.")
        FP_type = "known"
    elif fp_path != "":
        FP_type = "pre-trained"
        notify("use pre-trained forward kernel.")
    else:
        notify("ERROR: Need a forward kernel.")

    BP_type = "learned"

    # --------------------------------------------------------------------------
    psf_path = pathlib.Path(psf_path)
    fp_path = pathlib.Path(fp_path)
    bp_path = pathlib.Path(bp_path)
    # --------------------------------------------------------------------------
    # load parameters
    # forward projection model
    if FP_type == "pre-trained":
        parent_fp = pathlib.Path(fp_path).parent

        try:
            with open(pathlib.Path(parent_fp, "parameters.json")) as f:
                params_fp = json.load(f)
        except OSError as e:
            notify(str(e))
            return 0

        data_dim_fp = params_fp["data_dim"]
        ks_z_fp = params_fp["ks_z"]
        ks_xy_fp = params_fp["ks_xy"]

        if params_fp["model_name"] != "kernet_fp":
            notify("ERROR: The FP path does not point to a FP model.")
            return 0

        if data_dim != data_dim_fp:
            notify(
                f"ERROR: The dim of FP is {data_dim_fp}, but the dim of data is {data_dim}"
            )
            return 0

        if data_dim == 3:
            kernel_size_fp = (ks_z_fp, ks_xy_fp, ks_xy_fp)
        if data_dim == 2:
            kernel_size_fp = (ks_xy_fp,) * 2

    # backward projection model
    parent_bp = pathlib.Path(bp_path).parent

    try:
        with open(pathlib.Path(parent_bp, "parameters.json")) as f:
            params_bp = json.load(f)
    except OSError as e:
        notify(str(e))
        return 0

    data_dim_bp = params_bp["data_dim"]
    ks_z_bp = params_bp["ks_z"]
    ks_xy_bp = params_bp["ks_xy"]

    if params_bp["model_name"] != "kernet":
        notify("ERROR: The BP path does not point to a BP model.")
        return 0

    if data_dim != data_dim_bp:
        notify(
            f"ERROR: The dim of BP is {data_dim_bp}, but the dim of data is {data_dim}"
        )
        return 0

    if data_dim == 3:
        kernel_size_bp = (ks_z_bp, ks_xy_bp, ks_xy_bp)
    if data_dim == 2:
        kernel_size_bp = (ks_xy_bp,) * 2

    # --------------------------------------------------------------------------
    scale_factor = 1
    interpolation = True
    kernel_norm_fp = False
    kernel_norm_bp = True
    over_sampling = 2
    padding_mode = "reflect"
    if data_dim == 3:
        std_init = (4.0, 2.0, 2.0)
    if data_dim == 2:
        std_init = (2.0, 2.0)
    shared_bp = True
    conv_mode = "fft"

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    FP, BP = None, None
    # Forward Projection
    if FP_type == "pre-trained":
        notify("Use a pre-trained forward kernel.")
        notify(f"fp_path: {fp_path}")

        FP = kernelnet.ForwardProject(
            dim=data_dim,
            in_channels=in_channels,
            scale_factor=scale_factor,
            kernel_size=kernel_size_fp,
            std_init=std_init,
            init="gauss",
            kernel_norm=kernel_norm_fp,
            padding_mode=padding_mode,
            interpolation=interpolation,
            over_sampling=over_sampling,
            conv_mode=conv_mode,
        ).to(device)

        # ker_init = FP.conv.get_kernel().detach().numpy()[0, 0]
        FP.load_state_dict(
            torch.load(fp_path, map_location=device)["model_state_dict"]
        )
        FP.eval()

    # --------------------------------------------------------------------------
    if FP_type == "known":
        print("known PSF")
        PSF_true = io.imread(psf_path).astype(np.float32)
        ks = PSF_true.shape

        # check
        if len(ks) != data_dim:
            notify(
                f"ERROR: The dim of PSF is {len(ks)}, but the dim of data is {data_dim}"
            )

        weight = torch.tensor(PSF_true[None, None]).to(device=device)

        def padd_fp(x):
            if data_dim == 2:
                pad_size = (ks[-1] // 2, ks[-1] // 2, ks[-2] // 2, ks[-2] // 2)
            if data_dim == 3:
                pad_size = (
                    ks[-1] // 2,
                    ks[-1] // 2,
                    ks[-2] // 2,
                    ks[-2] // 2,
                    ks[-3] // 2,
                    ks[-3] // 2,
                )
            x_pad = torch.nn.functional.pad(
                input=x,
                pad=pad_size,
                mode=padding_mode,
            )
            return x_pad

        if conv_mode == "direct":

            def conv_fp(x):
                if data_dim == 2:
                    x_conv = torch.nn.functional.conv2d(
                        input=padd_fp(x), weight=weight, groups=in_channels
                    )
                if data_dim == 3:
                    x_conv = torch.nn.functional.conv3d(
                        input=padd_fp(x), weight=weight, groups=in_channels
                    )
                return x_conv

        if conv_mode == "fft":

            def conv_fp(x):
                return fft_conv(
                    signal=padd_fp(x), kernel=weight, groups=in_channels
                )

        def FP(x):
            if data_dim == 2:
                x_fp = torch.nn.functional.avg_pool2d(
                    conv_fp(x), kernel_size=scale_factor, stride=scale_factor
                )

            if data_dim == 3:
                x_fp = torch.nn.functional.avg_pool3d(
                    conv_fp(x), kernel_size=scale_factor, stride=scale_factor
                )
            return x_fp

        # ker_init = np.zeros_like(ker_FP)
        # The PSF now is known, setting the initial PSF as all zeros.
        ker_FP = weight.numpy()[0, 0]

    # --------------------------------------------------------------------------
    model = kernelnet.KernelNet(
        in_channels=in_channels,
        scale_factor=scale_factor,
        dim=data_dim,
        num_iter=num_iter,
        kernel_size_fp=kernel_size_fp,
        kernel_size_bp=kernel_size_bp,
        std_init=std_init,
        init="gauss",
        padding_mode=padding_mode,
        FP=FP,
        BP=BP,
        lam=0.0,
        return_inter=True,
        multi_out=False,
        over_sampling=over_sampling,
        kernel_norm=kernel_norm_bp,
        interpolation=interpolation,
        shared_bp=shared_bp,
        conv_mode=conv_mode,
        observer=observer,
    ).to(device)

    # --------------------------------------------------------------------------
    if BP_type == "learned":
        notify("use learned backward kernel.")
        model_path = bp_path
        notify(
            f"bp_path: {model_path}",
        )

        model.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"],
            strict=False,
        )
        model.eval()

        # get the learned BP kernel
        if shared_bp:
            ker_BP = model.BP.conv.get_kernel()[0, 0].detach().numpy()
        else:
            ker_BP = model.BP[0].conv.get_kernel()[0, 0].detach().numpy()

    notify(f"BP kernel shape: {ker_BP.shape}")

    if FP_type == "pre-trained":
        ker_FP = model.FP.conv.get_kernel()[0, 0].detach().numpy()
        notify(f"FP kernel shape: {ker_FP.shape}")

    # --------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------
    x = torch.from_numpy(img)[None, None]
    ts = time.time()
    y_pred_all = model(x)
    notify(
        f"Time : {time.time()-ts:.2f} s, each iteration: {(time.time()-ts)/num_iter:.2f} s."
    )
    y_pred_all = y_pred_all.cpu().detach().numpy()[:, 0, 0]
    y_pred = y_pred_all[-1]
    return y_pred


if __name__ == "__main__":

    img_path = "F:\\Datasets\\BioSR\\F-actin_Nonlinear\\raw_noise_9\\11.tif"
    psf_path = ""
    fp_path = "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\checkpoints\\forward_bs_1_lr_0.001_ks_1_31\\epoch_100.pt"
    bp_path = "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\checkpoints\\backward_bs_1_lr_1e-05_iter_2_ks_1_31\\epoch_100.pt"

    img = io.imread(img_path).astype(np.float32)
    y = predict(
        img,
        psf_path=psf_path,
        fp_path=fp_path,
        bp_path=bp_path,
        num_iter=2,
        observer=None,
    )
    print(y.shape)
