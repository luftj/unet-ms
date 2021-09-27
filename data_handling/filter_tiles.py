import os
import shutil
from PIL import Image
import argparse
import numpy as np

def filter_dir(indir, outdir, threshold, plot=False):
    masks_dir = indir + "/masks/"#"E:/data/usgs/100k/imgs/tiles/masks/"
    imgs_dir = indir + "/imgs/"#"E:/data/usgs/100k/imgs/tiles/imgs/"
    
    if not outdir:
        outdir = indir
    out_masks = outdir + "/masks/"
    out_imgs = outdir + "/imgs/"
    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_imgs, exist_ok=True)

    factors = []
    cut_factors = []

    for file in sorted(os.listdir(masks_dir)):#[:100]:
        if not os.path.isfile(masks_dir+file):
            continue
        
        img = Image.open(masks_dir + file)
        pixels = np.array(img)

        # calculate number of class pixels
        num_bg = pixels.size
        num_fg = np.count_nonzero(pixels)
        imgs_class_pixels = num_fg/num_bg
        img.close()
        print("%s: %0.3f" % (file, imgs_class_pixels))

        if imgs_class_pixels >= threshold:
            try:
                # use this tile -> move it to output
                    shutil.move(masks_dir + file, out_masks + file)
                    shutil.move(imgs_dir + file, out_imgs + file)
            except FileNotFoundError as e:
                print(e)
            cut_factors.append(imgs_class_pixels)
        else:
            # don't use this tile -> delete it
            os.remove(masks_dir + file)
            os.remove(imgs_dir + file)
            factors.append(imgs_class_pixels)

    avg_fact = sum(factors)/(len(factors) if len(factors)>0 else 1)
    print("avg fg/bg of filtered tiles:",avg_fact)
    print("number of tiles filtered: %d/%d"%(len(factors),len(factors)+len(cut_factors)))
    if plot:
        from matplotlib import pyplot as plt
        plt.hist(factors, bins=np.arange(0,max(factors),0.0005))
        plt.xticks(np.arange(0,max(factors),0.0005))
        plt.vlines(avg_fact, 0, len(factors)/10, colors="r")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('-o','--outdir', help='output directory')
    parser.add_argument('-t','--thresh', help='class threshold, i.e. proportion of foreground pixels to use a tile', type=float, default=0.003)
    parser.add_argument('--plot', help='plot class distribution', default=False, action="store_true")
    args = parser.parse_args()
    filter_dir(args.input, args.outdir, args.thresh, args.plot)