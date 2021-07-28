# unet-ms

![example segmentation](docs/title_fig.jpg)

Topographic map segmentation with u-nets.

WIP.

---

## Installation

Cloning this repo recursively to get the submodule containing the unet implementation with

```git clone --recursive git@github.com:luftj/unet-ms.git```

Requires
* Python3 (tested with 3.9.1)
* GDAL (for generating synthetic training data)

```$ python3 -m pip install -r requirements.txt``` (will also install requirements of the submodule)


## Usage

1. copy training data to Pytorch-UNet/data/imgs and Pytorch-UNet/data/masks respectively. We use 512x512px tiles.

2. train model as described [here](https://github.com/milesial/Pytorch-UNet)

3. segment a map image with `$ ./segment_image.sh [image path]`

## To Do
* fix path to checkpoints