#!/bin/bash

infile=$1
outdir="prediction_278"

mkdir $outdir
mkdir $outdir/tiles

python tile_images.py $infile $outdir/tiles/
cd Pytorch-UNet
python predict.py -m checkpoints/CP_epoch2.pth -i $(ls ../$outdir/tiles/imgs/*.tif)
cd ..

mkdir $outdir/pred_tiles
for f in $outdir/tiles/imgs/*_OUT* ; do
    filename=$(basename $f)
    mv "$f" "$outdir/pred_tiles/${filename/_OUT/}"
done

python merge_tiles.py ./$outdir/pred_tiles ./$outdir