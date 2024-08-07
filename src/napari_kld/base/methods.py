import napari_kld.base.deconvolution as dcv
import numpy as np
import skimage.io as skio
from napari.utils.notifications import show_info
from napari_kld.base.utils.dataset_utils import even2odd


def test_func(num_iter=1, observer=None):
    print("run function")
    for i in range(5):
        observer.progress(i + 1)
        observer.notify(f"deconv {0}")


def rl_deconv(
    img,
    psf_path,
    kernel_type="Traditional",
    num_iter=1,
    observer=None,
    alpha=0,
    beta=0,
    n=0,
):
    psf = skio.imread(psf_path).astype(np.float32)
    img = img.astype(np.float32)

    dim_psf = len(psf.shape)
    dim_img = len(img.shape)

    if dim_psf != dim_img:
        show_info(
            f"ERROR: the input is a {dim_img}D image, but the PSF is {dim_psf}D."
        )
        return 0

    psf = even2odd(psf)

    # --------------------------------------------------------------------------
    if kernel_type == "Traditional":
        DCV = dcv.Deconvolution(
            PSF=psf, bp_type="traditional", init="measured"
        )

    # --------------------------------------------------------------------------
    if kernel_type == "Gaussian":
        DCV = dcv.Deconvolution(PSF=psf, bp_type="gaussian", init="measured")

    # --------------------------------------------------------------------------
    if kernel_type == "Butterworth":
        # suggested parameters: beta=0.01, n=10, res_flag=1
        DCV = dcv.Deconvolution(
            PSF=psf,
            bp_type="butterworth",
            beta=beta,
            n=n,
            res_flag=1,
            init="measured",
        )

    # --------------------------------------------------------------------------
    if kernel_type == "WB":
        # suggested parameters: alpha=0.005, beta=0.1, n=10, res_flag=1
        DCV = dcv.Deconvolution(
            PSF=psf,
            bp_type="wiener-butterworth",
            alpha=alpha,
            beta=beta,
            n=n,
            res_flag=1,
            init="measured",
        )

    # --------------------------------------------------------------------------
    img_deconv = DCV.deconv(
        img, num_iter=num_iter, domain="fft", observer=observer
    )

    return img_deconv
