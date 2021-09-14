#!/bin/bash

# ulimit -n 3200 # avoid OSError: too many open files
# ulimit -n

data_path="E:/data/usgs/100k/"
out_path="E:/experiments/deepseg_models/"
quads="E:/data/usgs/indices/CellGrid_30X60Minute.json"
# data_path="/media/ecl2/DATA/jonas/usgs/100k_raw/"
# out_path="/media/ecl2/DATA/jonas/deepseg_models/"
# quads="/media/ecl2/DATA/jonas/usgs/CellGrid_30X60Minute.json"

sampled_data=$data_path/selected/
mkdir $sampled_data

# subsample number of input maps -- do train/test split
# allows full maps or tiles
train_data=$sampled_data/train/
test_data=$sampled_data/test/
# python random_subfiles.py $data_path $train_data --nomask -n 10 > $sampled_data/trainlist.txt
# python random_subfiles.py $data_path $test_data --nomask -n 3 -x $sampled_data/trainlist.txt > $sampled_data/testlist.txt

# # optional: rescale input images
# for file in $train_data/*.tif ; do
#     gdal_translate -outsize 5120 0 -r bilinear "$file" "${file/.tif/_rescaled.tif}"
#     rm "$file"
# done
# for file in $test_data/*.tif ; do
#     gdal_translate -outsize 5120 0 -r bilinear "$file" "${file/.tif/_rescaled.tif}"
#     rm "$file"
# done

# #create masks
# python synthesise_data.py $train_data $train_data $quads
# python synthesise_data.py $test_data $test_data $quads

#tile images and masks
tiled_train=$train_data/tiles/
# python tile_images.py $train_data/ $tiled_train -s 320
# python tile_images.py $train_data/ $tiled_train -s 320 -x 200 -y 200

# # remove empty masks + corresponding imgs
# python filter_tiles.py $tiled_train -t 0.01 --plot

# move data
mv persson_unet/data/ persson_unet/data_bak/
mkdir persson_unet/data/
mv $tiled_train persson_unet/data/
mv persson_unet/data/imgs persson_unet/data/train_images
mv persson_unet/data/masks persson_unet/data/train_masks
mkdir persson_unet/data/val_images
mkdir persson_unet/data/val_masks
shopt -s globstar
files=(/persson_unet/data/train_images/*)
for i in {1..50}; do
    mv "${files[RANDOM % ${#files[@]}]}" persson_unet/data/val_images
done
for f in persson_unet/data/val_images/*; do
    mv persson_unet/data/train_masks/$f persson_unet/data/val_masks/
done
exit
# run training
exp_no=37
echo "Exp# $exp_no"
python persson_unet/train_eth.py

mv persson_unet/data/ persson_unet/data_"$exp_no"_train/
# run som test maps
tail -n +2 $sampled_data/testlist.txt > $sampled_data/testlist2.txt
lines=$(cat $sampled_data/testlist2.txt)
while read -r file; do
    file=$(echo $file | tr -d '\r')
    echo "Testing with" $file
    
    #tile images and masks
    tiled_test=$test_data/tiles/
    python tile_images.py $test_data/ $tiled_test -s 320
    python tile_images.py $test_data/ $tiled_test -s 320 -x 200 -y 200

    python persson_unet/predict_eth.py

    mv persson_unet/predictions/pred_* persson_unet/predictions/pred_tiles/
    
    python merge_tiles.py persson_unet/predictions/pred_tiles/ "$out_path/$exp_no"
    rm -r persson_unet/predictions/
done <<< "$lines"

# calculate error scores of test map predictions and the corresponding masks
# touch scores_$exp_no_$param.txt
echo "scoring $exp_no..."
python score_predictions.py "$out_path/$exp_no" "$test_data" > scores_"$exp_no"_$param.txt
exp_no=$((exp_no + 1))

done # iterating over params
echo "done with all"
