# napari-kld

[![License](https://img.shields.io/pypi/l/napari-kld.svg?color=green)](https://github.com/qiqi-lu/napari-kld/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-kld.svg?color=green)](https://pypi.org/project/napari-kld)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-kld.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-kld)](https://napari-hub.org/plugins/napari-kld)

`napari-kld` is a `napari` plugin that implements kernel learning deconvolution algrotihm.

## Kernel Learning Deconvolution (KLD)

KLD is a rapid deconvolution algorithm for fluorescence microscopic image, which learns the forward and backward kernels in Richardson-Lucy Deconvolution (KLD) from paired low-/high-resolution images.

It only requires **one sample** to training the model, and **two iterations** to achieve a superior deconvolution performance compared to RLD and its variants using unmatched backward projection.

**This [napari] plugin was generated with [copier] using the [napari-plugin-template].*

## Installation

You can install `napari-kld` via [pip]:

    pip install napari-kld

## Instruction
This plugin includes two part:

- `RL Deconvolution` : Conventional RLD algorithm using different type of backward kernels (including matched backward kernel [`Traditional`] and unmatched backward kernels [`Guassian`, `Butterworth`, `Wiener-Butterworth (WB)`]). The forward kernel, i.e., point spread function (PSF), is required.

- `KL Deconvolution` : KLD using learned forward/backward kernels.

## RL Deconvolution

The conventional RLD using different type of backward kernels.

1. Open `napari`.

2. Load input low-resolution (LR) image: `File` > `Open File(s)` > `[choose the image to be deconvolved]` > `[the image will appear in the layer list of napari]`, such as the simulated image `"test\data\simulation\data_128_128_128_gauss_0.0_poiss_0_ratio_1.0\train\raw\.0.tif"`.

3. Choose the name of loaded image in `Input RAW data`, such as `"0"`.

4. Press `Choose` to choose a `PSF` correspongding to the loaded image, such as `"test/data/simulation/data_128_128_128_gauss_0.0_poiss_0_ratio_1.0/train/psf.tif"`.

5. Choose the type of backward kernel in `Method` combo box:

    - `Traditional` : the backward kernel is just the flip of forward kernel (i.e., PSF).
    - `Guassian` : Guassian-shaped backward kernel, thw FWHM of which is same as the forward kernel.
    - `Butterworth` : Butterworth-shaped backward kernel, which is constructed using Butterworth filter.
    - `WB` : WB-shaped backward kernel, which is constructed by combining Wiener and Butterworth filter.

6. Set the number of RL iterations `Iterations` and parameters of backward kernel*.

7. Press `run` button to do deconvolution.

8. Wait the `progress bar` to reach 100%.

9. The output deconved image will appear in the layer list named as `{name of input image}_deconv_{Method}_iter_{Iterations}`, such as `"0_deconv_traditional_iter_30"`.

**The adjustment of parameters of backward kernels should refer to the paper : Guo, M. et al. Rapid image deconvolution and multiview fusion for optical microscopy. Nat Biotechnol 38, 1337–1346 (2020).*

## KL Deconvolution

### When only with Point Spread Function (PSF)

When you only have a PSF to do deconvolution, you can train the model using simulated data.

1. Generate simulaiton data

2. Train the model under supervised mode

3. Apply the trained model on real data

#### Simulation data generation

1. Load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. Choose `Simulation` tab.

3. Choose the `Output directory` of the generated simulation data, such as `"napari-kld\src\napari_kld\_tests\work_directory\data\simulation"`.

4. Choose the `PSF directory` (only support 2D/3D PSF file save as .tif, axes = (z, y, x)), such as `"D:\GitHub\napari-kld\src\napari_kld\_tests\work_directory\data\simulation\PSF.tif"`.

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

1. Load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. Choose `Training` tab.

3. Choose `Data Directory ` (such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation/data/train"`) which saves the data used to train the model in should include:
    - A `gt` folder saves the GT images
    - A `raw` folder save the low-resolution raw input images with the same file name of GT images
    - A `train.txt` file saves all the file name used to train the model (does not need to list all the file in `gt`/`raw` folder).

4. Choose a `Output Directory` to save the model checkpoints, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation"`.

5. Choose `PSF Directory` of the PSF used to generate the data, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation/data/train/psf.tif"`. Then the `Forward Projection` group box will be invisible as we do not need to learn the forward kernel when we know the PSF. Just use the PSF as the froward kernel.

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

10. Training finished.

When the training finished, a checkpoints folder will be created in Ouput Directory such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/simulation/checkpoints"`.

The models is save in `/checkpoints` folder, which is named as `"backward_bs_{batch size}_lr_{learning rate}_iter_{num of RL iterations}_ks_{kernel size (z)}_{kernel size (xy)}"`, such as `"/checkpoints/backward_bs_1_lr_1e-06_iter_2_ks_1_31"`, consists of:

- A `log` folder saved the `Tensorboard` log, which can be open with `Tensorboard`.
- Many model checkpoints, named as `epoch_{epoch}.pt`.
- A `parameters.json` file saving the parameters used to training the model.

### When only with paired low-resolution (LR) image (as RAW) and high-resolution (HR) image (as GT)

When we only have paired LR image and HR image, we can first learned the forward kernel and then learn the backward kernel in a supervised strategy.

#### Training of Forward Projection

1. Load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. Choose `Training` tab.

3. Choose `Data Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real/train"`.

4. Choose `Output Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real"`.

5. `PSF Directory` is no required as the PSF is unknown.

6. Set parameter of parameters about data.
    - `Image Channel` : the channel of input image.
    - `Dimension` : dimension of input image.

7. In the `Forward Projection` box, set the parameters of training:
    - `Epoch` : number of epochs of training.
    - `Batch Size` : batch size of training data used during training.
    - `Kernel size (z, xy)` : the size of forward kernel to learned.
    - `Learning Rate` : learning rate of training.

8. Press `run` button. You can press the `stop` button to end the training.

9. Wait the `progress bar` to reach 100%.

10. Training finished.

After the training of forward projection, the results will be save in the `/checkpoints` folder in `Output Directory`, the model was named as `forward_bs_{batch size}_lr_{learning rate}_ks_{kernel size (z)}_{kernel size (xy)}`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real/checkpoints/forward_bs_1_lr_0.001_ks_1_31"`, which consists of:
- a `log` folder saved the `Tensorboard` log, which can be opened with `Tensorboard`.
- many model checkpoints, named as `epoch_{epoch}.pt`.
- a `parameters.json` file saving the parameters used to training the model.

#### Training of Backward Projection
After training of dorward projeciton, we can freeze the forward projeciton and then train the backward projeciton.

1. Load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. Choose `Training` tab.

3. Choose `Data Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real/train"`.

4. Choose `Output Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real"`.

5. `PSF Directory` is no required as the PSF is unknown.

6. set parameter of parameters about data.
    - `Image Channel` : the channel of input image.
    - `Dimension` : dimension of input image.

7. In the `Backward Projeciton` box, set parameters for the trianing of backward projeciton.

    - `Training strategy` : `supervised` training or `self-supervised` training. Set as `supervised`, as we have the GT images.
    - `Iteration (RL)` : The number of iterations of RL iterative procedure. Default: 2.
    - `Epoch` : The number fo epochs used to traing the model.
    - `Batch Size` : The batch size used to training the model.
    - `Kernel Size (z, xy)`: The size of backward kernel, `x` and `y` have the same size.
    - `FP directory` : the directory of the forward projeciton model, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real/checkpoints/forward_bs_1_lr_0.001_ks_1_31/epoch_100.pt"`
    - `Learning Rate` : The learning rate used to trianing the model.

8. Press `run` button. You can press the `stop` button to end the training.

9. Wait the `progress bar` to reach 100%.

10. Training finished.

When the training finishes, the results will be save in the `/checkpoints` folder in `Output Directory`, the model was named as `backward_bs_{batch size}_lr_{learning rate}_iter_{num of RL iterations}_ks_{kernel size (z)}_{kernel size (xy)}`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/checkpoints/backward_bs_1_lr_1e-05_iter_2_ks_1_31"`, which consists of:
- a `log` folder saved the `Tensorboard` log, which can be opened with `Tensorboard`.
- many model checkpoints, named as `epoch_{epoch}.pt`.
- a `parameters.json` file saving the parameters used to training the model.

Now we get the learned forward projection and backward projection.

### When only with LR image and corresponding PSF
When we only have LR image and its PSF, we can traing the backward projection through supervised training using simulation data as introduced above. The plugin also provide an alternative self-supervised training stratergy to learn the backward kernel.

1. load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. choose `Training` tab.

3. choose `Data Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real/train"`.

4. choose `Output Directory`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/data/real"`.

5. choose `PSF Directory`. Now the Forward Projection box will be invisiable.

6. set parameter of parameters about data.
    - `Image Channel` : the channel of input image.
    - `Dimension` : dimension of input image.

7. In the `Backward Projeciton` box, set parameters for the trianing of backward projeciton.

    - `Training strategy` : `supervised` training or `self-supervised` training. Set as `self-supervised`, as we do not have the GT images.
    - `Iteration (RL)` : The number of iterations of RL iterative procedure. Default: 2.
    - `Epoch` : The number fo epochs used to traing the model.
    - `Batch Size` : The batch size used to training the model.
    - `Kernel Size (z, xy)`: The size of backward kernel, `x` and `y` have the same size.
    - `FP directory` is not required as the PSF is known.
    - `Learning Rate` : The learning rate used to trianing the model.

8. Press `run` button. You can press the `stop` button to end the training.

9. Wait the `progress bar` to reach 100%.

10. Training finished.

When the training finishes, the results will be save in the `/checkpoints` folder in `Output Directory`, the model was named as `backward_bs_{batch size}_lr_{learning rate}_iter_{num of RL iterations}_ks_{kernel size (z)}_{kernel size (xy)}_ss`, such as `"D:/GitHub/napari-kld/src/napari_kld/_tests/work_directory/checkpoints/backward_bs_1_lr_1e-05_iter_2_ks_31_31_ss"`, which consists of:
- a `log` folder saved the `Tensorboard` log, which can be opened with `Tensorboard`.
- many model checkpoints, named as `epoch_{epoch}.pt`.
- a `parameters.json` file saving the parameters used to training the model.

### Prediction
Use the learned forward/backward kernel to do deconvolution.

1. load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. choose `Prediction` tab.

3. load raw input low-resolution image through `napari`: `File` > `Open File(s)` > `[choose the image to be deconvolved]` > `[the image will appear in the layer list of napari]`, such as `"F:\Datasets\BioSR\F-actin_Nonlinear\raw_noise_9\16.tif"`.

4. choose the loaded image in `Input RAW data` box, e.g., `16`.

5. if the PSF is known, choose the `PSF directory`.

6. if the PSF is unknown, choose the `Forward Projection` directory.*

7. choose the Backward Projeciton directory.

8. set the number of RL iterations at `Iterations (RL)`. Default: 2.

9. Press run to do deconvolution.

10. Wait the progress bar to reach 100%.

The deconvolved image will directly shwo in the `layer list` of `napari`, named as `"{input data name}_deconvo_iter_{number of RL iterations}"`, e.g., `"16_deconv_iter_2"`. You can save it as needed.


**If both the directories of PSF and Forward Projeciton is choosen, KLD will directly use the PSF selected.*


### Others
The `log` tab print the message during running.
Press `clean` button will clean all the text in the `log` box.

### Notice
- *The training time may be very long if we set the kernel size or the number of epoches too large, especially for 3D images.*

- *Now the plugin is runned on CPU. We have try to run the training on GPU, but the training time did not decrease (maybe it is because the covnlution based on FFT was not optimized on GPU). We are trying to make improvements.*


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
