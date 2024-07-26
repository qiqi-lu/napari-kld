import time

import napari_kld.base.deconvolution as dcv
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
    data_dim=3,
    in_channels=1,
    observer=None,
):

    def notify(value):
        print(value)
        if observer is not None:
            observer.notify(value)

    device = torch.device("cpu")

    if psf_path != "":
        FP_type = "known"
    elif psf_path == "" and fp_path != "":
        FP_type = "pre-trained"

    BP_type = "learned"

    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = [ker_size_fp, 31, 31]
    kernel_size_bp = [ker_size_bp, 25, 25]
    dim = 3

    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = (ker_size_fp,) * 2
    kernel_size_bp = (ker_size_bp,) * 2
    dim = 2

    # suffix_net = '_ss'
    # suffix_net = ""

    # ------------------------------------------------------------------------------
    num_iter_train = 2
    num_iter_test = num_iter_train + 0
    # ------------------------------------------------------------------------------
    scale_factor = 1
    interpolation = True
    kernel_norm_fp = False
    kernel_norm_bp = True
    over_sampling = 2
    padding_mode = "reflect"
    if dim == 3:
        std_init = [4.0, 2.0, 2.0]
    if dim == 2:
        std_init = [2.0, 2.0]
    shared_bp = True
    conv_mode = "fft"

    # ------------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------------
    FP, BP = None, None
    # Forward Projection
    print("-" * 80)
    if FP_type == "pre-trained":
        print("FP kernel (PSF) (Pre-trained)")
        print("model: ", fp_path)
        FP = kernelnet.ForwardProject(
            dim=dim,
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

    # ------------------------------------------------------------------------------
    if FP_type == "known":
        print("known PSF")
        PSF_true = io.imread(psf_path).astype(np.float32)
        ks = PSF_true.shape
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

    # ------------------------------------------------------------------------------
    # Backward Projection
    print("-" * 80)
    if BP_type == "known":
        print("BP kernel (Known)")

        def BP(x):
            return dcv.Convolution(
                PSF=ker_FP,
                x=x.detach().numpy()[0, 0],
                padding_mode=padding_mode,
                domain=conv_mode,
            )

        ker_BP = PSF_true

    # ------------------------------------------------------------------------------
    model = kernelnet.KernelNet(
        in_channels=in_channels,
        scale_factor=scale_factor,
        dim=dim,
        num_iter=num_iter_test,
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
    ).to(device)

    # ------------------------------------------------------------------------------
    if BP_type == "learned":
        print("BP kernel (Leanred)")
        model_path = bp_path
        print("Model: ", model_path)
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

    print("BP kernel shape:", ker_BP.shape)

    if FP_type == "pre-trained":
        # get the FP learned FP kernel
        ker_FP = model.FP.conv.get_kernel()[0, 0].detach().numpy()
        print("FP kernel shape:", ker_FP.shape)

    # --------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------
    def t2n(x):
        return x.cpu().detach().numpy()[0, 0]

    # prediciton
    x = img[None, None]
    ts = time.time()
    y_pred_all = model(x)
    notify(
        f"Time : {time.time()-ts:.2f} s, each iteration: {(time.time()-ts)/num_iter_test:.2f} s."
    )
    y_pred_all = y_pred_all.cpu().detach().numpy()[:, 0, 0]
    y_pred = y_pred_all[-1]
    return y_pred
