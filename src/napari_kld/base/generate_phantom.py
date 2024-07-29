"""
Simulation phantom generation.
"""

import os

import numpy as np
import skimage.io as io
import torch


def generate_phantom_3D(
    output_path,
    shape=(128, 128, 128),
    num_simulation=1,
    is_with_background=False,
):
    delta = 0.7
    Rsphere = 9
    Ldot = 9

    more_obj = (shape[0] // 128) * (shape[1] // 128) * (shape[2] // 128)
    n_spheres = 200 * more_obj
    n_ellipsoidal = 200 * more_obj
    n_dots = 50 * more_obj

    # create Gaussian filter
    Ggrid = range(-2, 2 + 1)
    [X, Y, Z] = np.meshgrid(Ggrid, Ggrid, Ggrid)
    # create Guassian mask
    GaussM = np.exp(-(X**2 + Y**2 + Z**2)) / (2 * delta**2)
    # normalize so thant total area (sum of all weights) is 1
    GaussM = GaussM / np.sum(GaussM)

    # spheroid
    for tt in range(num_simulation):
        print(f"simulation {tt}")
        A = np.zeros(shape=shape)
        Sz, Sy, Sx = A.shape

        rrange = np.fix(Rsphere / 2)
        xrange, yrange, zrange = (
            Sx - 2 * Rsphere,
            Sy - 2 * Rsphere,
            Sz - 2 * Rsphere,
        )  # avoid out of image range

        # ----------------------------------------------------------------------
        for _ in range(n_spheres):
            x = np.floor(xrange * np.random.rand() + Rsphere).astype(np.int8)
            y = np.floor(yrange * np.random.rand() + Rsphere).astype(np.int8)
            z = np.floor(zrange * np.random.rand() + Rsphere).astype(np.int8)

            r = np.floor(rrange * np.random.rand() + rrange).astype(np.int8)
            inten = 800 * np.random.rand() + 50

            for i in range(x - r, x + r + 1):
                for j in range(y - r, y + r + 1):
                    for k in range(z - r, z + r + 1):
                        if ((i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2) < r**2:
                            A[k, j, i] = inten

        # ----------------------------------------------------------------------
        for _ in range(n_ellipsoidal):
            x = np.floor(xrange * np.random.rand() + Rsphere).astype(np.int8)
            y = np.floor(yrange * np.random.rand() + Rsphere).astype(np.int8)
            z = np.floor(zrange * np.random.rand() + Rsphere).astype(np.int8)

            r1 = np.floor(rrange * np.random.rand() + rrange).astype(np.int8)
            r2 = np.floor(rrange * np.random.rand() + rrange).astype(np.int8)
            r3 = np.floor(rrange * np.random.rand() + rrange).astype(np.int8)

            inten = 800 * np.random.rand() + 50

            for i in range(x - r1, x + r1 + 1):
                for j in range(y - r2, y + r2 + 1):
                    for k in range(z - r3, z + r3 + 1):
                        if (
                            ((i - x) ** 2) / r1**2
                            + ((j - y) ** 2) / r2**2
                            + ((k - z) ** 2) / r3**2
                        ) <= 1.3 and (
                            ((i - x) ** 2) / r1**2
                            + ((j - y) ** 2) / r2**2
                            + ((k - z) ** 2) / r3**2
                        ) >= 0.8:
                            A[k, j, i] = inten

        # ----------------------------------------------------------------------
        dotrangex = Sx - Ldot - 1
        dotrangey = Sy - Ldot - 1
        dotrangez = Sz - Ldot - 1

        for _ in range(n_dots):
            x = np.floor((Sx - 3) * np.random.rand() + 1).astype(np.int8)
            y = np.floor((Sy - 3) * np.random.rand() + 1).astype(np.int8)
            z = np.floor((Sz - 3) * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50

            A[z : z + 2, y : y + 2, x : x + 2] = inten

        for _ in range(n_dots):
            x = np.floor(dotrangex * np.random.rand() + 1).astype(np.int8)
            y = np.floor((Sy - 3) * np.random.rand() + 1).astype(np.int8)
            z = np.floor((Sz - 3) * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50
            k = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)

            A[z : z + 2, y : y + 2, x : x + k + 1] = inten

        for _ in range(n_dots):
            x = np.floor((Sx - 3) * np.random.rand() + 1).astype(np.int8)
            y = np.floor(dotrangey * np.random.rand() + 1).astype(np.int8)
            z = np.floor((Sz - 3) * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50

            k = (np.floor(np.random.rand() * 9) + 1).astype(np.int8)

            A[z : z + 2, y : y + k + 1, x : x + 2] = (
                inten + 50 * np.random.rand()
            )

        for _ in range(n_dots):
            x = np.floor((Sx - 3) * np.random.rand() + 1).astype(np.int8)
            y = np.floor((Sy - 3) * np.random.rand() + 1).astype(np.int8)
            z = np.floor(dotrangez * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50
            k = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)

            A[z : z + k + 1, y : y + 2, x : x + 2] = inten

        for _ in range(n_dots):
            x = np.floor(dotrangex * np.random.rand() + 1).astype(np.int8)
            y = np.floor((Sy - 3) * np.random.rand() + 1).astype(np.int8)
            z = np.floor(dotrangez * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50
            k1 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            k2 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)

            A[z : z + k2 + 1, y : y + 2, x : x + k1 + 1] = inten

        for _ in range(n_dots):
            x = np.floor(dotrangex * np.random.rand() + 1).astype(np.int8)
            y = np.floor(dotrangey * np.random.rand() + 1).astype(np.int8)
            z = np.floor((Sz - 3) * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50

            k1 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            k2 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            A[z : z + 2, y : y + k2 + 1, x : x + k1 + 1] = inten

        for _ in range(n_dots):
            x = np.floor((Sx - 3) * np.random.rand() + 1).astype(np.int8)
            y = np.floor(dotrangey * np.random.rand() + 1).astype(np.int8)
            z = np.floor(dotrangez * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50
            k1 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            k2 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            A[z : z + k2 + 1, y : y + k1 + 1, x : x + 2] = inten

        for _ in range(n_dots):
            x = np.floor(dotrangex * np.random.rand() + 1).astype(np.int8)
            y = np.floor(dotrangey * np.random.rand() + 1).astype(np.int8)
            z = np.floor(dotrangez * np.random.rand() + 1).astype(np.int8)

            r = 1

            inten = 800 * np.random.rand() + 50
            k1 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            k2 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)
            k3 = (np.floor(np.random.rand() * Ldot) + 1).astype(np.int8)

            A[z : z + k3 + 1, y : y + k2 + 1, x : x + k1 + 1] = inten

        if is_with_background:
            A = A + 30

        A_torch = torch.Tensor(A)[None, None]
        GaussM_torch = torch.Tensor(GaussM)[None, None]

        A_conv = torch.nn.functional.conv3d(
            input=A_torch, weight=GaussM_torch, padding="same"
        )

        A_conv = A_conv.cpu().detach().numpy()
        A_conv = np.array(A_conv, dtype=np.float32)

        io.imsave(
            fname=os.path.join(output_path, f"{tt}.tif"),
            arr=A_conv,
            check_contrast=False,
        )


if __name__ == "__main__":
    import pathlib

    output_path = pathlib.Path(
        "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory"
    )
    shape = (128, 128, 128)
    num_simulation = 2

    generate_phantom_3D(
        output_path=output_path,
        shape=shape,
        num_simulation=num_simulation,
    )
