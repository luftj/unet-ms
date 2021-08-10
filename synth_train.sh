#!/bin/bash

# ulimit -n 3200 # avoid OSError: too many open files
# ulimit -n

data_path="E:/data/usgs/100k/"
out_path="E:/experiments/deepseg_models/"
quads="E:/data/usgs/indices/CellGrid_30X60Minute.json"

sampled_data=$data_path/selected/
mkdir $sampled_data

# subsample number of input maps -- do train/test split
# allows full maps or tiles
train_data=$sampled_data/train/
python random_subfiles.py $data_path $train_data --nomask -n 2 > $sampled_data/trainlist.txt
test_data=$sampled_data/test/
python random_subfiles.py $data_path $test_data --nomask -n 1 -x $sampled_data/trainlist.txt > $sampled_data/testlist.txt

# optional: rescale input images
for file in $train_data/*.tif ; do
    gdal_translate -outsize 5120 0 -r bilinear "$file" "${file/.tif/_rescaled.tif}"
    rm "$file"
done
for file in $test_data/*.tif ; do
    gdal_translate -outsize 5120 0 -r bilinear "$file" "${file/.tif/_rescaled.tif}"
    rm "$file"
done

#create masks
python synthesise_data.py $train_data $train_data $quads
python synthesise_data.py $test_data $test_data $quads

tile images and masks
python tile_images.py $data_path $data_path/tiles/ -s 1024
tiled_train=$train_data/tiles/
python tile_images.py $train_data/ $tiled_train -s 512
python tile_images.py $train_data $tiled_train -s 512 -x 256 -y 256

# remove empty masks + corresponding imgs
python filter_tiles.py $tiled_train -t 0.01 --plot

# todo: handle train param keys
model_path="/e/experiments/deepseg_models/"
exp_no=26
param_list=("1e-6" "1e-5")
for param in "${param_list[@]}" ; do
    echo "Exp# $exp_no: Training with param $param"
    echo python Pytorch-UNet/train.py $tiled_train "$model_path/checkpoints$exp_no" -l $param -e 100 -w 60 > log_train_$exp_no_$param.txt
    exp_no=$((exp_no + 1))
    
    # run som test maps
    tail -n +2 $sampled_data/testlist.txt > $sampled_data/testlist2.txt
    lines=$(cat $sampled_data/testlist2.txt)
    while read -r file; do
        file=$(echo $file | tr -d '\r')
        echo "Testing with" $file
        echo bash segment_image.sh $test_data/"${file/.tif/_rescaled.tif}" $model_path "$out_path" > log_test_$exp_no_$param.txt
    done <<< "$lines"
    
    # calculate error scores of test map predictions and the corresponding masks
    # touch scores_$exp_no_$param.txt
    python score_predictions.py "$out_path" "$test_data" > scores_$exp_no_$param.txt

done # iterating over params