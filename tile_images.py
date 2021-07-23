import os
import argparse
from PIL import Image

save_by_id = False

def tile_and_save(image_file_path, out_dir, tile_size=512, x_offset=0, y_offset=0):
    img = Image.open(image_file_path)
    width, height = img.size
    
    tile_idx = 0
    for x in range((width-x_offset)//tile_size): # this might not produce a tile for the right and lower borders -> mirror image to fill the last full size tile?
        for y in range((height-y_offset)//tile_size):
            x_pos = x*tile_size + x_offset
            y_pos = y*tile_size + y_offset
            box = (x_pos, y_pos, x_pos+tile_size, y_pos+tile_size)
            tile = img.crop(box)
            out_name = os.path.splitext(os.path.basename(image_file_path))[0]
            if "mask" in out_name:
                out_name = out_name.replace("_mask","")
                if save_by_id:
                    out_file = "%s/masks/%s_tile%03d.tif" % (out_dir, out_name, tile_idx)
                else:
                    out_file = "%s/masks/%s_tile%d-%d.tif" % (out_dir, out_name, x_pos, y_pos) 
                tile.save(out_file)
            else:
                if save_by_id:
                    out_file = "%s/imgs/%s_tile%03d.tif" % (out_dir, out_name, tile_idx)
                else:
                    out_file = "%s/imgs/%s_tile%d-%d.tif" % (out_dir, out_name, x_pos, y_pos)
                tile.save(out_file)
            tile_idx += 1
    print("dims:",x,y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input image or directory')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('x_offset', help='', type=int, default=0)
    parser.add_argument('y_offset', help='', type=int, default=0)
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
            tile_and_save(input_path+file, out_dir, x_offset=args.x_offset, y_offset=args.y_offset)
    else:
        # process single input file
        tile_and_save(input_path, out_dir, x_offset=args.x_offset, y_offset=args.y_offset)