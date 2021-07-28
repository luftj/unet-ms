#!/bin/bash

if [ $# -eq 0 ]; then
    echo "error: please provide an input file!"
    exit 1
fi

infile="$1"
exp_no="23"
epoch="148"
outdir="/e/experiments/deepseg_models/checkpoints$exp_no/scale_1/prediction_e""$epoch"_$(basename "$infile")
mkdir -p "$outdir"/tiles
python tile_images.py "$infile" "$outdir/tiles/"
cd Pytorch-UNet
# models=(E:/experiments/deepseg_models/checkpoints$exp_no/CP_epoch*.pth)
# model_path=${models[-1]}
model_path=E:/experiments/deepseg_models/checkpoints$exp_no/CP_epoch$epoch.pth
echo "using model at" $model_path

echo "starting prediction..."
for file in "$outdir"/tiles/imgs/*.tif ; do # todo: trouble with spaces in file names?
    python predict.py -m "$model_path" -i "$file" -s 1
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
# error-score feature-based (compare to indexing)

echo "done!"