#!/bin/bash

if [ $# -eq 0 ]; then
    echo "error: please provide an input file!"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "error: please provide model directory!"
    exit 1
fi

if [ $# -lt 3 ]; then
    echo "error: please provide output directory!"
    exit 1
fi

infile="$1"
#exp_no="23"
exp_no=$(ls -d -1 $2/checkpoints* | sed -e "s/[^0-9]*//" | sort -nr | head -n 1)
echo "Experiment no.:" $exp_no
#epoch="200"
epoch=$(ls -d -1 $2/checkpoints$exp_no/*.pth | sed -e "s/.*\///" | sed -e "s/[^0-9]*//g" | sort -nr | head -n 1)
echo "Epoch:" $epoch
#outdir="/e/experiments/deepseg_models/checkpoints$exp_no/test/prediction_e""$epoch"_$(basename "$infile")
outdir="$3/$exp_no/prediction_e""$epoch"_$(basename "$infile")
echo $outdir

mkdir -p "$outdir"/tiles
python tile_images.py "$infile" "$outdir/tiles/"

cd Pytorch-UNet
model_path="$2/checkpoints$exp_no/CP_epoch$epoch.pth"
echo "using model at" $model_path

echo "starting prediction..."
for file in "$outdir"/tiles/imgs/*.tif ; do # todo: trouble with spaces in file names?
    python predict.py -m "$model_path" -i "$file" #-s 1
done
# python predict.py -m $model_path -i $(ls $outdir/tiles/imgs/*.tif) #-s 1
cd ..

echo "moving tiles..."
mkdir "$outdir/pred_tiles"
for f in "$outdir"/tiles/imgs/*_OUT* ; do
    filename=$(basename "$f")
    filename=${filename/_OUT/}
    mv "$f" "$outdir"/pred_tiles/"$filename"
done

echo "merging tiles..."
python merge_tiles.py "$outdir/pred_tiles" "$outdir"

# todo: calculate error score, when ground truth is available
# error score pixel-wise
# problem: gt not really comparable, because lines on training with downscaling are thicker -> dilate gt or train on scale=1
# error-score feature-based (compare to indexing)

echo "done!"