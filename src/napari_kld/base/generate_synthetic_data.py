import json
import os

import napari_kld.base.deconvolution as dcv
import napari_kld.base.utils.dataset_utils as utils_data
import napari_kld.base.utils.evaluation as eva
import numpy as np
import skimage.io as io
from napari.utils.notifications import show_info
from napari_kld.base.generate_phantom import generate_phantom


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
    observer=None,
    **kwargs,
):
    def notify(value):
        if observer is not None:
            observer.notify(value)

    data_name = f"data_{image_shape[0]}_{image_shape[1]}_{image_shape[2]}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}"
    # --------------------------------------------------------------------------
    path_dataset_train = os.path.join(path_dataset, data_name, "train")
    path_dataset_gt = os.path.join(path_dataset_train, "gt")
    path_dataset_raw = os.path.join(path_dataset_train, "raw")

    for path in [path_dataset_gt, path_dataset_raw]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    notify("save to:")
    notify(path_dataset_gt)
    notify(path_dataset_raw)
    # --------------------------------------------------------------------------
    # load psf
    notify(f"load psf from: {path_psf}")
    psf = io.imread(path_psf).astype(np.float32)

    data_dim = 2 if image_shape[0] == 1 else 3

    if len(psf.shape) != data_dim:
        show_info(f"ERROR: {data_dim}D image but with {len(psf.shape)}D PSF.")
        return 0

    # --------------------------------------------------------------------------
    # generate ground truth phantom.
    generate_phantom(
        output_path=path_dataset_gt,
        shape=image_shape,
        num_simulation=num_simulation,
        is_with_background=False,
        observer=observer,
    )

    # --------------------------------------------------------------------------
    # read generated phantom data
    with open(os.path.join(path_dataset_train, "train.txt")) as f:
        text = f.read()
        name_list = text.split("\n")
        print(name_list)

    # --------------------------------------------------------------------------
    # generate raw image with blurring and noise.
    # interpolate psf with even shape to odd shape
    psf_odd = utils_data.even2odd(psf)
    notify(f"convert psf shape from {psf.shape} to {psf_odd.shape}.")

    # crop psf
    if data_dim == 3:
        psf_crop = utils_data.center_crop(psf_odd, size=psf_crop_shape)
    if data_dim == 2:
        psf_crop = utils_data.center_crop(
            psf_odd, size=(psf_crop_shape[1], psf_crop_shape[2])
        )
    notify(f"crop PSF from {psf_odd.shape} to a shape of {psf_crop.shape}")

    # save cropped psf
    io.imsave(
        os.path.join(path_dataset_train, "psf.tif"),
        arr=psf_crop,
        check_contrast=False,
    )

    # --------------------------------------------------------------------------
    img_gt_single = io.imread(os.path.join(path_dataset_gt, name_list[0]))
    img_gt_single = img_gt_single.astype(np.float32)
    notify(f"GT shape: {img_gt_single.shape}")

    # --------------------------------------------------------------------------
    # add blur and noise
    for name in name_list:
        if name == "":
            continue
        img_gt = io.imread(os.path.join(path_dataset_gt, name))

        # scale to control SNR
        img_gt = img_gt.astype(np.float32) * ratio
        img_blur = dcv.Convolution(
            img_gt, psf_crop, padding_mode="reflect", domain="fft"
        )

        # add noise
        img_blur_n = utils_data.add_mix_noise(
            img_blur,
            poisson=poisson,
            sigma_gauss=std_gauss,
            scale_factor=scale_factor,
        )

        # SNR
        notify(f"{name}, SNR: {eva.SNR(img_blur, img_blur_n)}")
        io.imsave(
            os.path.join(path_dataset_raw, name),
            arr=img_blur_n,
            check_contrast=False,
        )

    # --------------------------------------------------------------------------
    # save parameters
    params_dict = {
        "path_psf": path_psf,
        "image_shape": image_shape,
        "num_simulation": num_simulation,
        "psf_crop_shape": psf_crop_shape,
        "std_gauss": std_gauss,
        "poisson": poisson,
        "ratio": ratio,
        "scale_factor": scale_factor,
    }

    with open(os.path.join(path_dataset_train, "parameters.json"), "w") as f:
        f.write(json.dumps(params_dict))


if __name__ == "__main__":
    import pathlib

    output_path = pathlib.Path(
        "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\data\\simulation"
    )
    psf_path = pathlib.Path(
        "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\data\\simulation\\PSF.tif"
    )
    shape = (128, 128, 128)
    num_simulation = 2

    generate_simulation_data(
        path_dataset=output_path,
        path_psf=psf_path,
        num_simulation=num_simulation,
    )
