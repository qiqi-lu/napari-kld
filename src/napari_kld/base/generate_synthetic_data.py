import os

import napari_kld.base.deconvolution as dcv
import napari_kld.base.utils.dataset_utils as utils_data
import napari_kld.base.utils.evaluation as eva
import numpy as np
import skimage.io as io
from napari_kld.base.generate_phantom import generate_phantom_3D


def generate_simulation_data(
    path_dataset,
    path_psf,
    image_shape=(128, 128, 128),
    num_simulation=1,
    psf_crop_shape=None,
    std_gauss=0,
    poisson=1,
    ratio=1,
    scale_factor=1,
    observer=None
):
    def notify(value):
        print(value)
        if observer is not None:
            observer.notify(value)

    path_dataset_gt = os.path.join(path_dataset, "gt")
    path_dataset_raw = os.path.join(path_dataset, "raw")
    # --------------------------------------------------------------------------
    # generate ground truth phantom.
    generate_phantom_3D(
        output_path=path_dataset_gt,
        shape=image_shape,
        num_simulation=1,
        is_with_background=False,
    )

    # --------------------------------------------------------------------------
    # generate raw image with blurring and noise.

    # load psf
    notify("load psf from:", path_psf)
    psf = io.imread(path_psf).astype(np.float32)

    # interpolate psf with even shape to odd shape
    psf_odd = utils_data.even2odd(psf)
    notify(f"convert psf shape from {psf.shape} to {psf_odd.shape}.")

    # crop psf
    if psf_crop_shape is not None:
        size_crop = (np.minimum(psf_crop_shape[0], image_shape[0]),
            np.minimum(psf_crop_shape[1], image_shape[1]),
            np.minimum(psf_crop_shape[2], image_shape[2]))
        psf_crop = utils_data.center_crop(psf_odd, size=size_crop)
        print(f'crop PSF from {psf_odd.shape} to a shape of {psf_crop.shape}')

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
        os.path.join(path_dataset_raw, "psf.tif"),
        arr=psf,
        check_contrast=False,
    )
    # ------------------------------------------------------------------------------

    for i in [1]:
        data_gt = io.imread(os.path.join(path_dataset_gt, f"{i}.tif"))

        # scale to control SNR
        data_gt = data_gt.astype(np.float32) * ratio
        data_blur = dcv.Convolution(
            data_gt, psf_crop, padding_mode="reflect", domain="fft"
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
