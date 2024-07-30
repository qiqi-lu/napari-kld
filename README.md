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

### RL Deconvolution

### KL Deconvolution

#### Only with Point Spread Function (PSF)

When you only have a PSF to do deconvolution, you can train the model using simulated data.

1. generate simulaiton data

2. train the model under supervised mode

3. apply the trained model on real data

**simulation data generation**

1. load `napari-kld` plugin: `Plugins` > `Kernel Learning Deconvolution` > `KL Deconvolution`

2. choose `Simulation` tab.

3. choose the `Output directory` of the generated simulation data.

4. choose the `PSF directory` (only support 2D/3D PSF file save as .tif, axes = (z, y, x))








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
