#!/bin/bash

if [ $# -eq 0 ]; then
    echo "error: please provide an input file!"
    exit 1
fi

infile=$1
exp_no="14"
epoch="20"
outdir="/e/experiments/deepseg_models/checkpoints$exp_no/prediction_e""$epoch""_$(basename $infile)"

mkdir -p $outdir/tiles

python tile_images.py $infile $outdir/tiles/
cd Pytorch-UNet
# models=(E:/experiments/deepseg_models/checkpoints$exp_no/CP_epoch*.pth)
# model_path=${models[-1]}
model_path=E:/experiments/deepseg_models/checkpoints$exp_no/CP_epoch$epoch.pth
echo $model_path

python predict.py -m $model_path -i $(ls $outdir/tiles/imgs/*.tif)
cd ..

mkdir $outdir/pred_tiles
for f in $outdir/tiles/imgs/*_OUT* ; do
    filename=$(basename $f)
    mv "$f" "$outdir/pred_tiles/${filename/_OUT/}"
done

python merge_tiles.py $outdir/pred_tiles $outdir