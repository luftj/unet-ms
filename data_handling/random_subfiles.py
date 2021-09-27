import os
import shutil
import random
import argparse

def sample_random(input,outdir, num_images, exclude, nomask=False):
    if type(exclude) is list:
        exclude_list = []
    else:
        exclude_list = []
        with open(exclude, encoding="utf-8") as fr:
            for line in fr:
                exclude_list.append(line.strip())

    if "masks" in os.listdir(input):
        print("tiles")
        masks_dir = input + "/masks/"#"E:/data/usgs/100k/imgs/tiles/masks/"
        imgs_dir = input + "/imgs/"#"E:/data/usgs/100k/imgs/tiles/imgs/"
        os.makedirs(outdir+"/masks/", exist_ok=True)
        os.makedirs(outdir+"/imgs/", exist_ok=True)
        files = sorted([f for f in os.listdir(masks_dir) if not f in exclude_list])

        sampled_images = random.sample(files, num_images)

        for img in sampled_images:
            print(img)
            shutil.copyfile(masks_dir+img, outdir+"/masks/"+img)
            shutil.copyfile(imgs_dir+img, outdir+"/imgs/"+img)
    else:
        print("not tiles")
        os.makedirs(outdir, exist_ok=True)
        if nomask:
            files = sorted([f for f in os.listdir(input) if not "_mask" in f and not f in exclude_list])
        else:
            files = sorted([f for f in os.listdir(input) if "_mask" in f and not f in exclude_list])

        sampled_images = random.sample(files, num_images)

        for img in sampled_images:
            print(img)
            shutil.copyfile(input+img, outdir+"/"+img)
            if not nomask:
                shutil.copyfile(input+img.replace("_mask",""), outdir+"/"+img.replace("_mask",""))
    return sampled_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('-n','--num_images', help='how many images to sample', type=int, default=10)
    parser.add_argument('-x','--exclude', help='text file with images to exclude',default=[])
    parser.add_argument('--nomask', help='set this flags, if no masks should be copied', action="store_true", default=False)
    args = parser.parse_args()
    # num_images = 1000
    # outdir = outdir #"E:/data/usgs/100k/imgs/tiles/selected/"
    sample_random(args.input, args.outdir, args.num_images, args.exclude, args.nomask)
