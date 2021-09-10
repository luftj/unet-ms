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

if [ $# -eq 4 ]; then
    exp_no=$4
    echo "experiment number set manually"
else
    #exp_no="23"
    exp_no=$(ls -d -1 "$2"/checkpoints* |  grep -Eo "[0-9]*$" | sort -nr | head -n 1)
    echo "Experiment no.:" $exp_no
fi

infile="$1"
#epoch="200"
epoch=$(ls -d -1 $2/checkpoints$exp_no/*.pth | sed -e "s/.*\///" | sed -e "s/[^0-9]*//g" | sort -nr | head -n 1)
echo "Epoch:" $epoch
#outdir="/e/experiments/deepseg_models/checkpoints$exp_no/test/prediction_e""$epoch"_$(basename "$infile")
outdir="$3/$exp_no/prediction_e""$epoch"_$(basename "$infile")
echo "outdir: $outdir"

mkdir -p "$outdir"/tiles
python tile_images.py "$infile" "$outdir/tiles/"

cd Pytorch-UNet
model_path="$2/checkpoints$exp_no/CP_epoch$epoch.pth"
echo "using model at" $model_path

echo "starting prediction..."
for tile in "$outdir"/tiles/imgs/*.tif; do
        tiles+=("${tile}")
done

echo ${#tiles[@]} tiles found

batchsize=20
for ((f=0; f<"${#tiles[@]}"; f+=$batchsize)); do
        let g=f+$batchsize
        echo predicting tiles $f to $g...
        python predict.py -m "$model_path" -i "${tiles[@]:$f:$g}" -s 1 # allows spaces in filenames
done
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

echo "cleaning up..."
if compgen -G "$outdir/*.png" >/dev/null; then
    rm -r "$outdir/tiles"
    rm -r "$outdir/pred_tiles"
fi

# todo: calculate error score, when ground truth is available
# problem: gt not really comparable, because lines on training with downscaling are thicker -> dilate gt or train on scale=1
# calculate error scores of test map predictions and the corresponding masks
# touch scores_$exp_no_$param.txt
# python score_predictions.py "$out_path" "$test_data" > scores_$exp_no_$param.txt

echo "done!"
