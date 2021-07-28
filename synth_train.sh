#!/bin/bash

ulimit -n 3200 # avoid OSError: too many open files
ulimit -n

data_path="E:/data/usgs/100k/imgs/"
out_path="E:/experiments/deepseg_models/"
quads="E:/data/usgs/indices/CellGrid_30X60Minute.json"

python synthesise_data.py $data_path $data_path $quads

# subsample number of input maps -- do train/test split
# todo: allows full maps or tiles
sampled_data=$data_path/selected/
mkdir $sampled_data
python random_subfiles.py $data_path $sampled_data -n 10 > $sampled_data/list.txt

# tile images and masks
#python tile_images.py $data_path $data_path/tiles/ -s 1024
tiled_data=$sampled_data/tiles/
python tile_images.py $sampled_data $tiled_data -s 1024 -x 300 -y 300
#python tile_images.py $data_path $data_path/tiles/ -s 1024 -x 608 -y 57 

# remove empty masks + corresponding imgs
python filter_tiles.py $tiled_data "$tiled_data/cut" -t 0.003

#python Pytorch-UNet/train.py $tiled_data/cut $out_path -l 1e-6 -e 200 -w 10

# todo: run som test maps not in train set
# todo: calculate error scores