import os

import napari_kld.base.deconvolution as dcv
import napari_kld.base.utils.dataset_utils as utils_data
import napari_kld.base.utils.evaluation as eva
import numpy as np
import skimage.io as io


def generate_simulation_data(
    path_dataset,
    path_psf,
    image_shape=(128, 128, 128),
    psf_crop_shape=None,
    std_gauss=0,
    poisson=1,
    ratio=1,
    scale_factor=1,
):
    path_dataset_gt = os.path.join(path_dataset, "gt")
    path_dataset_raw = os.path.join(path_dataset, "raw")

    print("load PSF from:", path_psf)

    # --------------------------------------------------------------------------
    # load PSF
    PSF = io.imread(path_psf).astype(np.float32)
    PSF_odd = utils_data.even2odd(PSF)
    print(f"convert PSF shape from {PSF.shape} to {PSF_odd.shape}.")
    if psf_crop_shape is not None:
        PSF_crop = utils_data.center_crop(PSF_odd, size=psf_crop_shape)

    # ------------------------------------------------------------------------------
    data_gt_single = io.imread(os.path.join(path_dataset_gt, "1.tif"))
    data_gt_single = data_gt_single.astype(np.float32)
    print("GT shape:", data_gt_single.shape)

    std_gauss, poisson, ratio = 0.5, 1, 0.1
    scale_factor = 1

    # ------------------------------------------------------------------------------
    # save to
    if not os.path.exists(path_dataset_raw):
        os.makedirs(path_dataset_raw)
    print("save to:", path_dataset_raw)

    io.imsave(
        os.path.join(path_dataset_raw, "PSF.tif"),
        arr=PSF,
        check_contrast=False,
    )
    # ------------------------------------------------------------------------------

    for i in [1]:
        data_gt = io.imread(os.path.join(path_dataset_gt, f"{i}.tif"))

        # scale to control SNR
        data_gt = data_gt.astype(np.float32) * ratio
        data_blur = dcv.Convolution(
            data_gt, PSF_crop, padding_mode="reflect", domain="fft"
        )

        # add noise
        data_blur_n = utils_data.add_mix_noise(
            data_blur,
            poisson=poisson,
            sigma_gauss=std_gauss,
            scale_factor=scale_factor,
        )

        # SNR
        print(f"sample [{i}],", "SNR:", eva.SNR(data_blur, data_blur_n))
        io.imsave(
            os.path.join(path_dataset_raw, f"{i}.tif"),
            arr=data_blur_n,
            check_contrast=False,
        )
