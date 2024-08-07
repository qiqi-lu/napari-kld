"""
Simulation phantom generation.
Python version of the MATLAB code: "https://github.com/MeatyPlus/Richardson-Lucy-Net/blob/main/Phantom_generate/Phantom_generate.m"
"""

import os
import pathlib
from math import ceil

import numpy as np
import skimage.io as io
import torch


def generate_phantom(
    output_path,
    shape=(128, 128, 128),
    num_simulation=1,
    is_with_background=False,
    observer=None,
    **kwargs,
):
    delta = 0.7
    Rsphere = 9
    Ldot = 9

    if shape[0] == 1:
        print("make 2D image")
        data_dim = 2
        more_obj = (shape[1] / 128) * (shape[2] / 128)

        n_spheres = 5 * more_obj
        n_ellipsoidal = 5 * more_obj
        n_dots = 10 * more_obj

    elif shape[0] > 1:
        print("make 3D image")
        data_dim = 3
        more_obj = (shape[0] / 128) * (shape[1] / 128) * (shape[2] / 128)

        n_spheres = 200 * more_obj
        n_ellipsoidal = 200 * more_obj
        n_dots = 50 * more_obj

    n_spheres, n_ellipsoidal, n_dots = (
        np.maximum(ceil(n_spheres), 10),
        np.maximum(ceil(n_ellipsoidal), 10),
        np.maximum(ceil(n_dots), 10),
    )

    # create Gaussian filter
    Ggrid = range(-2, 2 + 1)
    if data_dim == 3:
        [Z, Y, X] = np.meshgrid(Ggrid, Ggrid, Ggrid)
        GaussM = np.exp(-(X**2 + Y**2 + Z**2)) / (2 * delta**2)

    if data_dim == 2:
        [Y, X] = np.meshgrid(Ggrid, Ggrid)
        GaussM = np.exp(-(X**2 + Y**2)) / (2 * delta**2)

    # normalize so thant total area (sum of all weights) is 1
    GaussM = GaussM / np.sum(GaussM)

    txt_path = pathlib.Path(output_path).parent
    if not os.path.exists(txt_path):
        os.makedirs(txt_path, exist_ok=True)

    # --------------------------------------------------------------------------
    with open(os.path.join(txt_path, "train.txt"), "w") as txt_file:
        # spheroid
        for tt in range(num_simulation):
            print(f"simulation {tt}")

            # ------------------------------------------------------------------
            # 3D image
            # ------------------------------------------------------------------
            if data_dim == 3:
                A = np.zeros(shape=shape)
                Sz, Sy, Sx = A.shape

                rrange = np.fix(Rsphere / 2)
                xrange, yrange, zrange = (
                    Sx - 2 * Rsphere,
                    Sy - 2 * Rsphere,
                    Sz - 2 * Rsphere,
                )  # avoid out of image range

                # --------------------------------------------------------------
                for _ in range(n_spheres):
                    x = np.floor(xrange * np.random.rand() + Rsphere)
                    y = np.floor(yrange * np.random.rand() + Rsphere)
                    z = np.floor(zrange * np.random.rand() + Rsphere)

                    r = np.floor(rrange * np.random.rand() + rrange)
                    inten = 800 * np.random.rand() + 50

                    x, y, z, r = int(x), int(y), int(z), int(r)
                    for i in range(x - r, x + r + 1):
                        for j in range(y - r, y + r + 1):
                            for k in range(z - r, z + r + 1):
                                if (
                                    (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2
                                ) < r**2 and (
                                    0 <= i < Sx and 0 <= j < Sy and 0 <= k < Sz
                                ):
                                    A[k, j, i] = inten

                # --------------------------------------------------------------
                for _ in range(n_ellipsoidal):
                    x = np.floor(xrange * np.random.rand() + Rsphere)
                    y = np.floor(yrange * np.random.rand() + Rsphere)
                    z = np.floor(zrange * np.random.rand() + Rsphere)

                    r1 = np.floor(rrange * np.random.rand() + rrange)
                    r2 = np.floor(rrange * np.random.rand() + rrange)
                    r3 = np.floor(rrange * np.random.rand() + rrange)

                    x, y, z, r1, r2, r3 = (
                        int(x),
                        int(y),
                        int(z),
                        int(r1),
                        int(r2),
                        int(r3),
                    )

                    inten = 800 * np.random.rand() + 50

                    for i in range(x - r1, x + r1 + 1):
                        for j in range(y - r2, y + r2 + 1):
                            for k in range(z - r3, z + r3 + 1):
                                if (
                                    (
                                        ((i - x) ** 2) / r1**2
                                        + ((j - y) ** 2) / r2**2
                                        + ((k - z) ** 2) / r3**2
                                    )
                                    <= 1.3
                                    and (
                                        ((i - x) ** 2) / r1**2
                                        + ((j - y) ** 2) / r2**2
                                        + ((k - z) ** 2) / r3**2
                                    )
                                    >= 0.8
                                    and (
                                        0 <= i < Sx
                                        and 0 <= j < Sy
                                        and 0 <= k < Sz
                                    )
                                ):
                                    A[k, j, i] = inten

                # --------------------------------------------------------------
                dotrangex = Sx - Ldot - 1
                dotrangey = Sy - Ldot - 1
                dotrangez = Sz - Ldot - 1

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)
                    z = np.floor((Sz - 3) * np.random.rand() + 1)

                    x, y, z = int(x), int(y), int(z)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    A[z : z + 2, y : y + 2, x : x + 2] = inten

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)
                    z = np.floor((Sz - 3) * np.random.rand() + 1)

                    r = 1

                    inten = 800 * np.random.rand() + 50
                    k = np.floor(np.random.rand() * Ldot) + 1

                    x, y, z, k = int(x), int(y), int(z), int(k)

                    A[z : z + 2, y : y + 2, x : x + k + 1] = inten

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)
                    z = np.floor((Sz - 3) * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k = np.floor(np.random.rand() * 9) + 1

                    x, y, z, k = int(x), int(y), int(z), int(k)

                    A[z : z + 2, y : y + k + 1, x : x + 2] = (
                        inten + 50 * np.random.rand()
                    )

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)
                    z = np.floor(dotrangez * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k = np.floor(np.random.rand() * Ldot) + 1

                    x, y, z, k = int(x), int(y), int(z), int(k)
                    A[z : z + k + 1, y : y + 2, x : x + 2] = inten

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)
                    z = np.floor(dotrangez * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k1 = np.floor(np.random.rand() * Ldot) + 1
                    k2 = np.floor(np.random.rand() * Ldot) + 1

                    x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)

                    A[z : z + k2 + 1, y : y + 2, x : x + k1 + 1] = inten

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)
                    z = np.floor((Sz - 3) * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k1 = np.floor(np.random.rand() * Ldot) + 1
                    k2 = np.floor(np.random.rand() * Ldot) + 1
                    x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)

                    A[z : z + 2, y : y + k2 + 1, x : x + k1 + 1] = inten

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)
                    z = np.floor(dotrangez * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k1 = np.floor(np.random.rand() * Ldot) + 1
                    k2 = np.floor(np.random.rand() * Ldot) + 1

                    x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)
                    A[z : z + k2 + 1, y : y + k1 + 1, x : x + 2] = inten

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)
                    z = np.floor(dotrangez * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k1 = np.floor(np.random.rand() * Ldot) + 1
                    k2 = np.floor(np.random.rand() * Ldot) + 1
                    k3 = np.floor(np.random.rand() * Ldot) + 1

                    x, y, z, k1, k2, k3 = (
                        int(x),
                        int(y),
                        int(z),
                        int(k1),
                        int(k2),
                        int(k3),
                    )

                    A[z : z + k3 + 1, y : y + k2 + 1, x : x + k1 + 1] = inten

                if is_with_background:
                    A = A + 30

                A_torch = torch.Tensor(A)[None, None]
                GaussM_torch = torch.Tensor(GaussM)[None, None]

                A_conv = torch.nn.functional.conv3d(
                    input=A_torch, weight=GaussM_torch, padding="same"
                )

            # ------------------------------------------------------------------
            # 2D image
            # ------------------------------------------------------------------
            if data_dim == 2:
                A = np.zeros(shape=(shape[1], shape[2]))
                Sy, Sx = A.shape

                rrange = np.fix(Rsphere / 2)
                xrange, yrange = (
                    Sx - 2 * Rsphere,
                    Sy - 2 * Rsphere,
                )  # avoid out of image range

                # --------------------------------------------------------------
                for _ in range(n_spheres):
                    x = np.floor(xrange * np.random.rand() + Rsphere)
                    y = np.floor(yrange * np.random.rand() + Rsphere)

                    r = np.floor(rrange * np.random.rand() + rrange)
                    inten = 800 * np.random.rand() + 50

                    x, y, r = int(x), int(y), int(r)

                    for i in range(x - r, x + r + 1):
                        for j in range(y - r, y + r + 1):
                            if ((i - x) ** 2 + (j - y) ** 2) < r**2:
                                A[j, i] = inten

                # --------------------------------------------------------------
                for _ in range(n_ellipsoidal):
                    x = np.floor(xrange * np.random.rand() + Rsphere)
                    y = np.floor(yrange * np.random.rand() + Rsphere)

                    r1 = np.floor(rrange * np.random.rand() + rrange)
                    r2 = np.floor(rrange * np.random.rand() + rrange)

                    inten = 800 * np.random.rand() + 50

                    x, y, r1, r2 = int(x), int(y), int(r1), int(r2)

                    for i in range(x - r1, x + r1 + 1):
                        for j in range(y - r2, y + r2 + 1):
                            if (
                                ((i - x) ** 2) / r1**2 + ((j - y) ** 2) / r2**2
                            ) <= 1.3 and (
                                ((i - x) ** 2) / r1**2 + ((j - y) ** 2) / r2**2
                            ) >= 0.8:
                                A[j, i] = inten

                # --------------------------------------------------------------
                dotrangex = Sx - Ldot - 1
                dotrangey = Sy - Ldot - 1

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50

                    x, y = int(x), int(y)
                    A[y : y + 2, x : x + 2] = inten

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor((Sy - 3) * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k = np.floor(np.random.rand() * Ldot) + 1

                    x, y, k = int(x), int(y), int(k)
                    A[y : y + 2, x : x + k + 1] = inten

                for _ in range(n_dots):
                    x = np.floor((Sx - 3) * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k = np.floor(np.random.rand() * 9) + 1
                    x, y, k = int(x), int(y), int(k)

                    A[y : y + k + 1, x : x + 2] = inten + 50 * np.random.rand()

                for _ in range(n_dots):
                    x = np.floor(dotrangex * np.random.rand() + 1)
                    y = np.floor(dotrangey * np.random.rand() + 1)

                    r = 1
                    inten = 800 * np.random.rand() + 50
                    k1 = np.floor(np.random.rand() * Ldot) + 1
                    k2 = np.floor(np.random.rand() * Ldot) + 1
                    x, y, k1, k2 = int(x), int(y), int(k1), int(k2)
                    A[y : y + k2 + 1, x : x + k1 + 1] = inten

                if is_with_background:
                    A = A + 30

                A_torch = torch.Tensor(A)[None, None]
                GaussM_torch = torch.Tensor(GaussM)[None, None]

                A_conv = torch.nn.functional.conv2d(
                    input=A_torch, weight=GaussM_torch, padding="same"
                )

            # ------------------------------------------------------------------
            A_conv = A_conv.cpu().detach().numpy()
            A_conv = np.array(A_conv, dtype=np.float32)[0, 0]

            io.imsave(
                fname=os.path.join(output_path, f"{tt}.tif"),
                arr=A_conv,
                check_contrast=False,
            )

            txt_file.write(f"{tt}.tif\n")
            if observer is not None:
                observer.progress(tt + 1)


if __name__ == "__main__":
    import pathlib

    output_path = pathlib.Path(
        "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory"
    )
    shape = (128, 128, 128)
    shape = (10, 64, 64)
    num_simulation = 2

    generate_phantom(
        output_path=output_path,
        shape=shape,
        num_simulation=num_simulation,
    )
