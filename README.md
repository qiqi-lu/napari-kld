# napari-kld

[![License](https://img.shields.io/pypi/l/napari-kld.svg?color=green)](https://github.com/qiqi-lu/napari-kld/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-kld.svg?color=green)](https://pypi.org/project/napari-kld)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-kld.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-kld)](https://napari-hub.org/plugins/napari-kld)

<font color=red> **This plugin is not completed.** </font>

----
### Kernel Learning Deconvolution (KLD)

Kernel learning deconvolution  is a rapid deconvolution algorithm for fluorescence microscopic image, which learns the forward and backward kernels in Richardson-Lucy Deconvolution (KLD) from paired low-/high-resolution images.

It only requires one sample to training the model, and two iteration to achieve a superior deconvolution perfromance compared to RLD and its variants using unmatched backward projection.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-kld` via [pip]:

    pip install napari-kld


## Instruction
This plugin includes two part:

- `RL Deconvolution` : Conventional RLD algorithm using different type of backward kernels (including matched backward kernel [`Traditional`] and unmatched backward kernels [`Guassian`, `Butterworth`, `Wiener-Butterworth`]). The forward kernel (i.e., PSF) must to be known.

- `KL Deconvolution` : KLD using learned forward/backward kernels.

## RL Deconvolution

## KL Deconvolution

### When only with Point Spread Function (PSF)

When you only have a PSF to do deconvolution, you can train the model using simulated data.

1. generate simulaiton data

2. train the model under supervised mode

3. apply the trained model on real data

#### Simulation data generation

1. load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. choose `Simulation` tab.

3. choose the `Output directory` of the generated simulation data, such as `"napari-kld\src\napari_kld\_tests\work_directory\data\simulation"`.

4. choose the `PSF directory` (only support 2D/3D PSF file save as .tif, axes = (z, y, x)), such as `"D:\GitHub\napari-kld\src\napari_kld\_tests\work_directory\data\simulation\PSF.tif"`.

5. Adjust the parameters as needed.
    - `Image shape` : the shape of simulated image, when `z=1`, 2D images will be generated.

    - `PSF crop` : when the input PSF is too large, you can crop the PSF to acuqire a smaller PSF, which is normalized after cropping. All the PSF will be converted to have an odd shape.

    - `Num of simulation` : number of generated images.

    - `Gaussian (std)` : The standard deviation of Gaussian noise added in the generated low-resolution raw images. The mean of Gaussian noise = 0. Default: 0 (i.e., without gaussian noise).

    - `Poisson` : whether to add Poisson noise, if `True`, make the `Enable` checked.

    - `Ratio` : the ratio multiply on ground truth (GT) image to control the level of Poisson noise, thus

    $$ RAW = Possion((GT \cdot Ratio)\times PSF) + Gaussian $$

    - `Scale factor` : downsampling scale factor. Default: 1.

6. Press `run` button.

7. Wait the `progress bar` to reach 100%.

The generated simulation data will be save in `Output directory`, such as: `"D:\GitHub\napari-kld\src\napari_kld\_tests\work_directory\data\simulation\data\train"`

- `"data\train\gt"` save the GT images.
- `"data\train\raw"` save the RAW images with blur and noise.
- `"data\train\parameters.json"` is the dictionary of parameter used to generate the simulated data.
- `"data\train\psf.tif"` is the psf used in generation of simulation data (as the original PSF may be cropped)
- `"data\train\train.txt` save all the image used to train the model.

*You may need to adjust the noise level in the image accordding to the real acuqired data.*

After you generate simulated data, you can use them to train the model.

#### Training with known PSF and simulated data

The simulated data should be those generated using the known PSF.

1. load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. choose `Training` tab.

3. choose `Data Directory ` (such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation/data/train"`) which saves the data used to train the model in should include:
    - a `gt` folder saves the GT images
    - a `raw` folder save the low-resolution raw input images with the same file name of GT images
    - a `train.txt` file saves all the file name used to train the model (does not need to list all the file in `gt`/`raw` folder).

4. choose a `Output Directory` to save the model checkpoints, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation"`.

5. choose `PSF Directory` of the PSF used to generate the data, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation/data/train/psf.tif"`. Then the `Forward Projection` group box will be invisible as we do not need to learn the forward kernel when we know the PSF. Just use the PSF as the froward kernel.

6. Set the `Image Channels` and the `Dimension` of input data.

7. Then set parameters to learn the backward kernel.

    - `Training strategy` : `supervised` training or `self-supervised` training. Set as `supervised`, as we have the GT images.
    - `Iteration (RL)` : The number of iterations of RL iterative procedure. Default: 2.
    - `Epoch` : The number fo epochs used to traing the model.
    - `Batch Size` : The batch size used to training the model.
    - `Kernel Size (z, xy)`: The size of backward kernel, `x` and `y` have the same size.
    - `FP directory` : the directory of the forward projeciton model. Here, it is empty (i.e., `""`) as the PSF is known.
    - `Learning Rate` : The learning rate used to trianing the model.

8. Press `run` button. You can press the `stop` button to end the training.

9. Wait the `progress bar` to reach 100%.
















## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

MIT LICENSE

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
