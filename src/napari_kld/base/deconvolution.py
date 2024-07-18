import napari_kld.base.back_projector as back_projector
import numpy as np
import torch
import tqdm
from fft_conv_pytorch import fft_conv


# ------------------------------------------------------------------------------
def align_size(img, size, padValue=0):
    dim = len(img.shape)

    if dim == 3:
        Nz_1, Ny_1, Nx_1 = img.shape
        Nz_2, Ny_2, Nx_2 = size
        Nz = np.maximum(Nz_1, Nz_2)
        Ny, Nx = np.maximum(Ny_1, Ny_2), np.maximum(Nx_1, Nx_2)

        imgTemp = np.ones(shape=(Nz, Ny, Nx)) * padValue
        imgTemp = imgTemp.astype(img.dtype)

        Nz_o = int(np.round((Nz - Nz_1) / 2))
        Ny_o, Nx_o = int(np.round((Ny - Ny_1) / 2)), int(
            np.round((Nx - Nx_1) / 2)
        )
        imgTemp[Nz_o : Nz_o + Nz_1, Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = (
            img
        )

        Nz_o = int(np.round((Nz - Nz_2) / 2))
        Ny_o, Nx_o = int(np.round((Ny - Ny_2) / 2)), int(
            np.round((Nx - Nx_2) / 2)
        )
        img2 = imgTemp[
            Nz_o : Nz_o + Nz_2, Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2
        ]

    if dim == 2:
        Ny_1, Nx_1 = img.shape
        Ny_2, Nx_2 = size
        Ny, Nx = np.maximum(Ny_1, Ny_2), np.maximum(Nx_1, Nx_2)

        imgTemp = np.ones(shape=(Ny, Nx)) * padValue
        imgTemp = imgTemp.astype(img.dtype)

        Ny_o, Nx_o = int(np.round((Ny - Ny_1) / 2)), int(
            np.round((Nx - Nx_1) / 2)
        )
        imgTemp[Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = img

        Ny_o, Nx_o = int(np.round((Ny - Ny_2) / 2)), int(
            np.round((Nx - Nx_2) / 2)
        )
        img2 = imgTemp[Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2]

    return img2


def ConvFFT3_S(inVol, OTF):
    outVol = np.fft.ifftn(np.fft.fftn(inVol) * OTF)
    return outVol.real


def Convolution(x, PSF, padding_mode="reflect", domain="direct"):
    ks = PSF.shape
    dim = len(ks)
    PSF, x = torch.tensor(PSF[None, None]), torch.tensor(x[None, None])
    PSF = torch.round(PSF, decimals=16)

    if dim == 3:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(
                ks[2] // 2,
                ks[2] // 2,
                ks[1] // 2,
                ks[1] // 2,
                ks[0] // 2,
                ks[0] // 2,
            ),
            mode=padding_mode,
        )
        if domain == "direct":
            x_conv = torch.nn.functional.conv3d(
                input=x_pad, weight=PSF, groups=1
            )
        if domain == "fft":
            x_conv = fft_conv(signal=x_pad, kernel=PSF, groups=1)

    if dim == 2:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(ks[1] // 2, ks[1] // 2, ks[0] // 2, ks[0] // 2),
            mode=padding_mode,
        )
        if domain == "direct":
            x_conv = torch.nn.functional.conv2d(
                input=x_pad, weight=PSF, groups=1
            )
        if domain == "fft":
            x_conv = fft_conv(signal=x_pad, kernel=PSF, groups=1)

    out = x_conv.numpy()[0, 0]
    return out


class Deconvolution:
    def __init__(
        self,
        PSF,
        bp_type="traditional",
        alpha=0.05,
        beta=1,
        n=10,
        res_flag=1,
        i_res=(2.44, 2.44, 10),
        init="measured",
        metrics=None,
        padding_mode="reflect",
    ):

        self.padding_mode = padding_mode
        self.bp_type = bp_type
        # forward PSF
        self.PSF1 = PSF / np.sum(PSF)
        # backward PSF
        self.PSF2, _ = back_projector.BackProjector(
            PSF_fp=PSF,
            bp_type=bp_type,
            alpha=alpha,
            beta=beta,
            n=n,
            res_flag=res_flag,
            i_res=i_res,
        )
        self.PSF2 = self.PSF2 / np.sum(self.PSF2)

        self.smallValue = 0.001
        self.init = init

        self.metrics = metrics
        self.metrics_value = []

        self.OTF_fp = None
        self.OTF_bp = None

    def measure(self, stack):
        self.metrics_value.append(self.metrics(stack))

    def deconv(self, stack, num_iter, domain="fft", observer=None):
        self.metrics_value = []
        print("=" * 80)
        print("Convolution in", domain, "domain.")
        print("BP Type:", self.bp_type)
        print("PSF shape (FP/BP):", self.PSF1.shape, self.PSF2.shape)

        size = stack.shape
        PSF_fp = align_size(self.PSF1, size)
        PSF_bp = align_size(self.PSF2, size)

        self.OTF_fp = np.fft.fftn(np.fft.ifftshift(PSF_fp))
        self.OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))

        stack = np.maximum(stack, self.smallValue)

        # initialization
        print(">> Initialization : ", self.init)
        if self.init == "constant":
            stack_estimate = np.ones(shape=stack.shape) * np.mean(stack)
        else:
            stack_estimate = stack

        # iterations
        pbar = tqdm.tqdm(desc="Deconvolution", total=num_iter, ncols=80)

        if self.metrics is not None:
            self.measure(stack_estimate)

        for i in range(num_iter):
            fp = Convolution(
                stack_estimate,
                PSF=self.PSF1,
                padding_mode=self.padding_mode,
                domain=domain,
            )
            dv = stack / (fp + self.smallValue)
            bp = Convolution(
                dv,
                PSF=self.PSF2,
                padding_mode=self.padding_mode,
                domain=domain,
            )
            stack_estimate = stack_estimate * bp
            stack_estimate = np.maximum(stack_estimate, self.smallValue)

            if self.metrics is not None:
                self.measure(stack_estimate)

            # observe iterations
            if observer is not None:
                observer.progress(i + 1)
                observer.notify("Iteration")
            pbar.update(1)
        pbar.close()
        return stack_estimate

    def get_metrics(self):
        return np.stack(self.metrics_value)
