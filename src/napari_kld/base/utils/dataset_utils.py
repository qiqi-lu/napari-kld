import os

import numpy as np
import pydicom
import skimage.io as skio
import torch
from skimage import io, transform
from torch.utils.data import Dataset


def get_image_shape(path):
    img = skio.imread(path)
    return img.shape


def interp(x, ps_xy=1, ps_z=1):
    x = np.array(x, dtype=np.float32)
    num_dim = len(x.shape)

    if num_dim == 3:
        z_scale = ps_z / ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(
            x, scale_factor=(z_scale, 1, 1), mode="nearest"
        )
        x = x.numpy()[0, 0]

    if num_dim == 2:
        z_scale = ps_z / ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(
            x, scale_factor=(z_scale, 1), mode="nearest"
        )
        x = x.numpy()[0, 0]
    return x


def gauss_kernel_1d(shape=3, std=1.0):
    x = torch.linspace(start=0, end=shape - 1, steps=shape)
    x_center = (shape - 1) / 2

    g = torch.exp(-((x - x_center) ** 2 / (2.0 * std**2)))
    g = g / torch.sum(g)  # shape = 3
    return g


def gauss_kernel_2d(shape=(3, 3), std=(1.0, 1.0), pixel_size=(1.0, 1.0)):
    x_data, y_data = np.mgrid[0 : shape[0], 0 : shape[1]]
    x_center, y_center = (shape[0] - 1) / 2, (shape[1] - 1) / 2

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)

    g = torch.exp(
        -(
            ((x - x_center) * pixel_size[0]) ** 2 / (2.0 * std[0] ** 2)
            + ((y - y_center) * pixel_size[1]) ** 2 / (2.0 * std[1] ** 2)
        )
    )
    g = g / torch.sum(g)  # shape = [3, 3]
    return g


def gauss_kernel_3d(
    shape=(3, 3, 3), std=(1.0, 1.0, 1.0), pixel_size=(1.0, 1.0, 1.0)
):
    x_data, y_data, z_data = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    x_center, y_center, z_center = (
        (shape[0] - 1) / 2,
        (shape[1] - 1) / 2,
        (shape[2] - 1) / 2,
    )
    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(
        -(
            ((x - x_center) * pixel_size[0]) ** 2 / (2.0 * std[0] ** 2)
            + ((y - y_center) * pixel_size[1]) ** 2 / (2.0 * std[1] ** 2)
            + ((z - z_center) * pixel_size[2]) ** 2 / (2.0 * std[2] ** 2)
        )
    )
    g = g / torch.sum(g)  # shape = [3, 3, 3]
    return g


def padding_kernel(x, y):
    dim = len(y.shape)
    if dim == 3:
        i_x, j_x, k_x = x.shape
        i_y, j_y, k_y = y.shape
        if (j_x <= j_y) & (i_x <= i_y):
            x = np.pad(
                x,
                pad_width=(
                    ((i_y - i_x) // 2,) * 2,
                    ((j_y - j_x) // 2,) * 2,
                    ((k_y - k_x) // 2,) * 2,
                ),
            )
    if dim == 2:
        j_x, k_x = x.shape
        j_y, k_y = y.shape
        if j_x <= j_y:
            x = np.pad(
                x, pad_width=(((j_y - j_x) // 2,) * 2, ((k_y - k_x) // 2,) * 2)
            )
    return x


def ave_pooling(x, scale_factor=1):
    """Average pooling for 2D/3D image."""
    x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2:
        x = torch.nn.functional.avg_pool2d(
            x[None, None], kernel_size=scale_factor
        )
    if len(x.shape) == 3:
        x = torch.nn.functional.avg_pool3d(
            x[None, None], kernel_size=scale_factor
        )
    return x.numpy()[0, 0]


def add_mix_noise(x, poisson=0, sigma_gauss=0, scale_factor=1):
    """Add Poisson and Gaussian noise."""
    x = np.maximum(x, 0.0)
    # add poisson noise
    x_poi = np.random.poisson(lam=x) if poisson == 1 else x

    # downsampling
    if scale_factor > 1:
        x_poi = ave_pooling(x_poi, scale_factor=scale_factor)

    # add gaussian noise
    if sigma_gauss > 0:
        max_signal = np.max(x_poi)
        x_poi_norm = x_poi / max_signal
        x_poi_gaus = x_poi_norm + np.random.normal(
            loc=0, scale=sigma_gauss / max_signal, size=x_poi_norm.shape
        )
        x_n = x_poi_gaus * max_signal
    else:
        x_n = x_poi

    return x_n.astype(np.float32)


def fft_n(kernel, s=None):
    kernel_fft = np.abs(np.fft.fftshift(np.fft.fftn(kernel, s=s)))
    return kernel_fft


def center_crop(x, size):
    """Crop the center region of the 3D image x."""
    dim = len(x.shape)
    if dim == 3:
        Nz, Ny, Nx = x.shape
        out = x[
            Nz // 2 - size[0] // 2 : Nz // 2 + size[0] // 2 + 1,
            Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
            Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
        ]
    if dim == 2:
        Ny, Nx = x.shape
        out = x[
            Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
            Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
        ]
    return out


def even2odd(x):
    """Convert the image x to a odd-shape image."""
    dim = len(x.shape)
    assert dim in [2, 3], "Only 2D or 3D image are supported."
    if dim == 3:
        i, j, k = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1
        if k % 2 == 0:
            k = k - 1
        x = torch.tensor(x)
        x_inter = torch.nn.functional.interpolate(
            x[None, None], size=(i, j, k), mode="trilinear"
        )
    if dim == 2:
        i, j = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1
        x = torch.tensor(x)
        x_inter = torch.nn.functional.interpolate(
            x[None, None], size=(i, j), mode="bilinear"
        )
    x_inter = x_inter / x_inter.sum()
    return x_inter.numpy()[0, 0]


def percentile_norm(x, p_low=0, p_high=100):
    """percentile-based normalization."""
    xmax, xmin = np.percentile(x, p_high), np.percentile(x, p_low)
    x = (x - xmin) / (xmax - xmin)
    x = np.clip(x, a_min=0.0, a_max=1.0)
    return x


def linear_transform(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    x_linear = b_1 * x + b_0
    return x_linear


def read_image(img_path, normalization=False, data_range=None):
    """
    Read image.
    Args:
    - img_path (str): Image path.
    - normalization (bool): Normalize data into (0,1).
    - data_range (tuple): (min, max) value of data.
    """
    # check file type, get extension of file
    _, ext = os.path.splitext(img_path)

    # DICOM data
    if ext == ".dcm":
        img_dcm = pydicom.dcmread(img_path)
        img = img_dcm.pixel_array
        img = img.astype(np.float32)

    # TIFF data
    if ext == ".tif":
        img = io.imread(img_path)

    if len(img.shape) == 2 or len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    # Image normalization
    if normalization:
        if data_range is None:
            img_max, img_min = img.max(), img.min()
            img = (img - img_min) / (img_max - img_min)
        if type(data_range) == tuple:
            img = (img - data_range[0]) / (data_range[1] - data_range[0])

    return img.astype(np.float32)


class SRDataset(Dataset):
    """
    Super-resolution dataset used to get low-resolution and hig-resolution data.
    Args:
    - hr_root_path (str): root path for high-resolution data.
    - lr_root_path (str): root path for  low-resolution data.
    - hr_txt_file_path (str): path of file saving path of high-resolution data.
    - lr_txt_file_path (str): path of file saving path of low-resolution data.
    - id_range (tuple): extract part of the data.
                        Default: None, all the data in dataset.
    - transform (bool): data transformation. Default: None.
    - normalization (tuple[bool]): whether to normalize the data
                    when read image (lr, hr). Default: (False, False).
    """

    def __init__(
        self,
        hr_root_path,
        lr_root_path,
        hr_txt_file_path,
        lr_txt_file_path,
        id_range=None,
        transform=None,
        normalization=(False, False),
    ):
        super().__init__()
        self.hr_root_path = hr_root_path
        self.lr_root_path = lr_root_path
        self.transform = transform
        self.normalization = normalization

        with open(lr_txt_file_path) as f:
            self.file_names_lr = f.read().splitlines()
        with open(hr_txt_file_path) as f:
            self.file_names_hr = f.read().splitlines()

        if id_range is not None:
            self.data_size = len(self.file_names_lr)
            self.file_names_lr = self.file_names_lr[id_range[0] : id_range[1]]
            self.file_names_hr = self.file_names_hr[id_range[0] : id_range[1]]

            print(
                f"DATASET: use only part of data set. ({len(self.file_names_lr)}|{self.data_size})"
            )
        else:
            print("DATASET: use all the data list in the train.txt")

    def __len__(self):
        return len(self.file_names_lr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path_lr = os.path.join(self.lr_root_path, self.file_names_lr[idx])
        img_path_hr = os.path.join(self.hr_root_path, self.file_names_hr[idx])

        image_lr = read_image(img_path_lr, normalization=self.normalization[0])
        image_hr = read_image(img_path_hr, normalization=self.normalization[1])

        if self.transform is not None:
            image_lr = self.transform(image_lr)
            image_hr = self.transform(image_hr)

        # scale = np.percentile(image_hr, 95)
        # return {'lr': torch.tensor(image_lr/scale), 'hr': torch.tensor(image_hr/scale)}
        return {"lr": torch.tensor(image_lr), "hr": torch.tensor(image_hr)}


class Rescale:
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_lr, image_hr = sample["lr"], sample["hr"]

        h, w = image_lr.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image_lr_new = transform.resize(image_lr, (new_h, new_w))

        return {"lr": image_lr_new, "hr": image_hr}


class ToNumpy:
    """
    Convert pytorch tensor into numpy array, and shift the channel axis to the last axis.
    Args:
    - tensor (torch tensor): input tensor.
    """

    def __call__(self, tensor):
        img = tensor.cpu().detach().numpy()
        # move the chennel axis to the last dimension.
        if len(img.shape) == 4:
            img = np.transpose(img, axes=(0, 2, 3, 1))
        if len(img.shape) == 5:
            img = np.transpose(img, axes=(0, 2, 3, 4, 1))
        return img


def tensor2rgb(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    x = (x * 255.0).to(torch.uint8)
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5:
        x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x


def tensor2gray(x):
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5:
        x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x


if __name__ == "__main__":
    import pathlib

    path = pathlib.Path(
        "D:\\GitHub\\napari-kld\\src\\napari_kld\\_tests\\work_directory\\data\\simulation\\PSF.tif"
    )
    print(get_image_shape(path))
