import napari_kld.base.deconvolution as dcv


def rl_deconv(
    img,
    psf,
    kernel_type="Traditional",
    num_iter=1,
    observer=None,
    super_params=None,
):
    """
    Args:
    - super_params: dict
        'sigma': sigma used in Gaussian kernels.
        'alpha': alpha used in Butterworth and WB backward kernels.

    """
    if kernel_type == "Traditional":
        DCV = dcv.Deconvolution(
            PSF=psf, bp_type="traditional", init="measured"
        )
        img_deconv = DCV.deconv(
            img, num_iter=num_iter, domain="fft", observer=observer
        )

    if kernel_type == "Gaussian":
        DCV = dcv.Deconvolution(PSF=psf, bp_type="gaussian", init="measured")
        img_deconv = DCV.deconv(
            img, num_iter=num_iter, domain="fft", observer=observer
        )

    if kernel_type == "Butterworth":
        DCV = dcv.Deconvolution(
            PSF=psf,
            bp_type="butterworth",
            beta=0.01,
            n=10,
            res_flag=1,
            init="measured",
        )
        img_deconv = DCV.deconv(
            img, num_iter=num_iter, domain="fft", observer=observer
        )

    if kernel_type == "WB":
        DCV = dcv.Deconvolution(
            PSF=psf,
            bp_type="wiener-butterworth",
            alpha=0.005,
            beta=0.1,
            n=10,
            res_flag=1,
            init="measured",
        )
        img_deconv = DCV.deconv(
            img, num_iter=num_iter, domain="fft", observer=observer
        )

    return img_deconv
