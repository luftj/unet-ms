import re
import os
import argparse
from PIL import Image

def merge_image(name, dims, output_dir):
    tile_idx = 0
    full_img=None
    for x in range(dims[0]+1):
        for y in range(dims[1]+1):
            print("merging %d,%d" % (x,y))
            path ="%s/%s_tile%03d.png" % (in_dir, name, tile_idx)
            print(path)
            cur_img = Image.open(path)
            if not full_img:
                tile_size = cur_img.size
                # set canvas size
                full_img = Image.new("RGB", (tile_size[0] * dims[0], tile_size[1] * dims[1]), (0,0,0))
            tile_pos = (tile_size[0]*x, tile_size[1]*y)
            full_img.paste(cur_img, tile_pos)
            if x == dims[0] and y == dims[1]:
                last_size = cur_img.size
            tile_idx += 1
    full_img = full_img.crop([0,0,tile_size[0]*x+last_size[0],tile_size[1]*y+last_size[1]])
    full_img.save(output_dir+"/"+name+".png", "PNG")
    
def merge_image_xy(name, positions, output_dir):
    tile_idx = 0
    full_img=None
    for position in positions:
        x,y = position
        print("merging %d,%d" % (x,y))
        path ="%s/%s_tile%d-%d.png" % (in_dir, name, x, y)
        print(path)
        cur_img = Image.open(path)
        if not full_img:
            tile_size = cur_img.size
            # set canvas size
            full_img=Image.new("RGB", (max(list(zip(*positions))[0]) + tile_size[0], max(list(zip(*positions))[1]) + tile_size[1]), (0,0,0))
        full_img.paste(cur_img, (x,y))
        tile_idx += 1
    first_offset = (min(list(zip(*positions))[0]), min(list(zip(*positions))[1]))
    print("cropping to",[*first_offset, *full_img.size])
    full_img = full_img.crop([*first_offset, *full_img.size])
    full_img.save(output_dir+"/"+name+".png", "PNG")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    # e.g. python merge_tiles.py E:/data/deutsches_reich/train/tiles3/masks ./prediction
    in_dir = args.input#"E:/data/deutsches_reich/train/tiles3/masks"

    images_dict  = {}

    for file in os.listdir(in_dir):
        file, ext = os.path.splitext(file)
        name, tile_no = file.split("_tile")
        print(name, tile_no, ext)

        if not name in images_dict:
            images_dict[name] = []
        images_dict[name].append(tile_no)
    
    for name, tiles in images_dict.items():
        if "-" in tiles[0]:
            positions = [tuple(map(int,x.split("-"))) for x in tiles]
            print(positions)
            merge_image_xy(name, positions, output_dir=args.outdir)
        else:
            merge_image(name, dims=(13,11), output_dir=args.outdir)