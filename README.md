# Image-Based Reconstruction
A 3D reconstruction pipeline making use of both implicit and explicit surface representations

# Installation

Only Linux is supported (Tested on Ubuntu 22.04)

Run the following command to create a conda envrionment

```
conda env create -f environment.yml
```

Then please follow the instructions in https://pytorch.org/ to install a recent version of PyTorch with cuda support (tested on PyTorch 1.11 w/ cuda 11.7)

After installing PyTorch, please install [Tiny Cuda Neural Network](https://github.com/NVlabs/tiny-cuda-nn) by running the following command

```
pip install -r requirements.txt
```

Then run `build_psdr.sh` to build enoki and psdr-cuda after installing the required dependencies listed in the [official doc of psdr](https://psdr-cuda.readthedocs.io/en/latest/core_compile.html)

```
source build_psdr.sh /path/to/optix/sdk /path/to/python/include/dir
```

Finally please follow the instructions in [large-steps-pytorch](https://github.com/rgl-epfl/large-steps-pytorch/tree/581866f52a137c21d0e27931ffdcf81576633158) to install the submodule

Run the `download_dataset_nerf_synthetic.py` script to download the nerf synthetic dataset and put the `nerf_synthetic` folder under the `./data` directory. Run the `preprocess_nerf_synthetic()` function in `util.py` before optimizing

After each stage of optmization you will have to run the `postprocess.py` to post-process the optimization results in order to do run the next stage or to get the optimized mesh. This is due to some subtle bug that makes it difficult to dump the results directly after optimization.

Please run `setpath.sh` to add some necessary envrionment variables before the first run

```
source setpath.sh
```

Run `optimize_dmtet.py` to run the first stage of optimization. The results are in `output/scene_name/` where `scene_name` is the `name` field in the config file.

Run `optimize_mesh.py` to run the second stage after running the first stage and post-processing the results. The results of the second stage are in `output/scene_name/` where `scene_name` is the `name` field in the config file.