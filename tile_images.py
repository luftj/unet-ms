import os
import argparse
from PIL import Image

def tile_and_save(image_file_path, out_dir, tile_size = 512):
    img = Image.open(image_file_path)
    width, height = img.size
    
    tile_idx = 0
    for x in range(width//tile_size): # this might not produce a tile for the right and lower borders -> mirror image to fill the last full size tile?
        for y in range(height//tile_size):
            box = (x*tile_size,y*tile_size,x*tile_size+tile_size,y*tile_size+tile_size)
            # print(box)
            tile = img.crop(box)
            out_name = os.path.splitext(os.path.basename(image_file_path))[0]
            if "mask" in out_name:
                out_name = out_name.replace("_mask","")
                out_file = "%s/masks/%s_tile%03d.tif" % (out_dir,out_name,tile_idx) 
                tile.save(out_file)
            else:
                out_file = "%s/imgs/%s_tile%03d.tif" % (out_dir,out_name,tile_idx) 
                tile.save(out_file)
            tile_idx += 1
    print("dims:",x,y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input image or directory')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    # e.g. python tile_images.py /e/data/deutsches_reich/train/ /e/data/deutsches_reich/train/tiles3/

    input_path = args.input #"E:/data/deutsches_reich/train/"
    out_dir = args.outdir #"E:/data/deutsches_reich/train/tiles3/"
    os.makedirs(out_dir+"/masks", exist_ok=True)
    os.makedirs(out_dir+"/imgs", exist_ok=True)
    file_ext = [".tif",".tiff",".jpg",".jpeg",".png",".bmp"]

    if os.path.isdir(input_path):
        # process all image files in directory
        for file in os.listdir(input_path):
            if not os.path.isfile(input_path+file):
                continue
            if not os.path.splitext(file)[-1] in file_ext:
                print(os.path.splitext(file)[-1])
                continue
            tile_and_save(input_path+file, out_dir)
    else:
        # process singel input file
        tile_and_save(input_path, out_dir)