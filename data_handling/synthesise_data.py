import json
import os
import subprocess
import re
import shutil
from PIL import Image
from pyproj import Transformer
import osm
import argparse
import config

from data_handling.utils import is_valid_map, is_valid_mask

def make_worldfile(inputfile, bbox, border, transform_sheet_to_out):
    """ create a worldfile for a warped map image given bounding box GCPS
    bbox as [left_x, bottom_y, right_x, top_y]
    border as [min_col, min_row, max_col, max_row]
    """
    minxy = transform_sheet_to_out.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
    maxxy = transform_sheet_to_out.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
    bbox = minxy+maxxy

    pixel_width = (bbox[2]-bbox[0])/(border[2]-border[0])
    pixel_height = (bbox[3]-bbox[1])/(border[3]-border[1])

    left_edge = bbox[0] - (border[0]-0.5) * pixel_width # subtract half pixel to get to the center of topleft corner
    top_edge = bbox[3] - (border[3]-0.5) * pixel_height # subtract half pixel to get to the center of topleft corner

    outputfile = os.path.splitext(inputfile)[0]+".tfw"
    with open(outputfile,"w") as fw:
        fw.write("%.20f\n" % pixel_width)
        fw.write("0.0"+"\n")
        fw.write("0.0"+"\n")
        fw.write("%.20f\n" % pixel_height)
        fw.write("%.20f\n" % left_edge)
        fw.write("%.20f\n" % top_edge)
    
    print("saved worldfile to: %s" % outputfile)

def get_coords_from_raster(georef_image_path, point):
    import rasterio # leave this in here, to avoid proj_db errors?
    dataset = rasterio.open(georef_image_path)

    latlong = dataset.transform * point

    return latlong

def get_pixel_from_raster(georef_image_path, lonlat):
    import rasterio # leave this in here, to avoid proj_db errors?

    dataset = rasterio.open(georef_image_path)
    # print(config.proj_sheets)
    # print(dataset.crs)
    tr = Transformer.from_proj(config.proj_sheets, dataset.crs, skip_equivalent=True, always_xy=True)
    lonlat = tr.transform(*lonlat)
    xy = dataset.index(*lonlat)[::-1]

    return xy

def synthesise_all(maps_dir, imgs_dir, quads_path):
    synthesise_list(maps_dir, imgs_dir, quads_path, file_list=os.listdir(maps_dir))

def synthesise_list(maps_dir, imgs_dir, quads_path, file_list):
    # load quadrangles data
    with open(quads_path, encoding="utf8") as fr:
        quadrangles = { f["properties"][config.quadrangles_key] : f["geometry"]["coordinates"][0] for f in json.load(fr)["features"]}
    # iterate over maps
    valid_ext = [".tif"]
    for file in file_list: # ["PA_Bradford_170431_1980_100000_geo.tif"]:#
        if not os.path.splitext(file)[-1] in valid_ext:
            continue
        # map_name = file.split("_")[1] # this is only for usgs100
        map_name = os.path.splitext(file)[0] #this si for kdro100
        # map_name = map_name.replace("St ","Saint ")
        print(file)
        print(map_name)
        # ret = osm.proj_map = subprocess.run("gdalinfo "+ maps_dir+"/"+file)
        result = subprocess.check_output("gdalinfo \""+ maps_dir+"/"+file+"\"", shell=True)
        # print(result.decode("ascii"))
        # print()
        crs = re.findall(r"(?<=Coordinate System is:).*(?=Data axis|Origin)",result.decode("ascii"), flags=re.MULTILINE+re.DOTALL)
        # osm.proj_map = crs[0].replace("\r","").replace("\n","")

        osm.transform_osm_to_map = Transformer.from_proj(config.proj_osm, osm.proj_map, skip_equivalent=True, always_xy=True)
        osm.transform_sheet_to_map = Transformer.from_proj(config.proj_sheets, osm.proj_map, skip_equivalent=True, always_xy=True)
        osm.transform_sheet_to_osm = Transformer.from_proj(config.proj_sheets, config.proj_osm, skip_equivalent=True, always_xy=True)
        # print(result.decode("ascii"))
        # p = subprocess.Popen(["gdalinfo "+ maps_dir+"/"+file], stdout=subprocess.PIPE)
        # out = p.stdout.read()
        # print(out)
        # print(type(out))
        
        # find correspoding quadrangle
        if not map_name in quadrangles:
            print("no quad found for map %s at %s! Skipping." % (map_name, file))
            continue
        quad = quadrangles[map_name]
        # print("quad",quad)

        # get px size of original map
        map_img = Image.open(maps_dir + "/" + file)
        map_size = map_img.size # x,y
        # print(map_size)

        # find graticule/neatline corner pixel coordinates in georef maps
        margin = []
        for coord in quad[:4]:
            # print(coord)
            transf_coord = get_pixel_from_raster(maps_dir + "/" + file, coord)
            # print(transf_coord)
            margin.append(transf_coord)
        print("neatline points",margin)
        xs, ys = list(zip(*margin))
        margin_left = min(xs)
        # margin_right = xs[-1]#max(xs) # HACK: trapezoid usgs100?
        margin_right = sorted(xs)[-2] # maybe like this?
        # margin_right = max(xs)
        margin_top = min(ys)
        margin_bottom = max(ys)
        print(margin_left,    margin_right,    margin_top,    margin_bottom   )
        # download osm data
        xs, ys = list(zip(*quad))
        bbox = [min(xs),min(ys),max(xs),max(ys)]
        print("map bbox", bbox)

        osm_features = osm.get_from_osm(bbox)
        img_size = (margin_right-margin_left, margin_bottom-margin_top)
        print(map_size, img_size)
        img = osm.paint_features(osm_features, bbox, img_size=img_size)
        if img is None:
            print("error in painting OSM mask, skipping")
            continue
        
        # pad OSM with margin
        full_img=Image.new("L", map_size, 0)
        # print("left neatline:",(margin_left))
        # print("right neatline:",(img_size[0]+margin_left))
        full_img.paste(Image.fromarray(img), (margin_left,margin_top,margin_right,margin_bottom))

        # from matplotlib import pyplot as plt
        # plt.imshow(full_img)
        # plt.show()

        # save osm mask
        # os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)
        full_img.save(imgs_dir + "/" + os.path.splitext(file)[0]+"_mask.tif")
        bbox = [min(xs),max(ys),max(xs),min(ys)]
        make_worldfile(imgs_dir + "/" + os.path.splitext(file)[0]+"_mask.tif",bbox,[margin_left, margin_top, margin_right, margin_bottom], osm.transform_sheet_to_map)
        if maps_dir != imgs_dir:
            shutil.copy(maps_dir + "/" + file, imgs_dir + "/" + file)
        
        # tile images after

def synthesise_maps_if_necessary(maps_dir, out_dir,set_name=""):
    test_masks = list(filter(is_valid_mask, os.listdir(out_dir)))
    test_maps = list(filter(is_valid_map, os.listdir(out_dir)))
    test_maps_without_masks = [map_i for map_i in test_maps if not map_i.replace(".","_mask.") in test_masks]
    if len(test_maps_without_masks) > 0: # todo: or if OSM params changed
        # if no: synthesise data
        print("synthesising %d missing %s masks..." % (len(test_maps_without_masks), set_name))
        synthesise_list(out_dir, out_dir, maps_dir, file_list=test_maps_without_masks)
    else:
        print("all %s masks present" % set_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('quads', help='quadrangles geojson file')
    args = parser.parse_args()

    maps_dir = args.input#"E:/data/usgs/100k"
    imgs_dir = args.outdir#+"/imgs/"#"E:/data/usgs/100k/imgs"
    quads_path = args.quads
    # masks_dir = args.outdir+"/masks/"#"E:/data/usgs/100k/masks"
    # quadrangles_file = "E:/data/usgs/indices/CellGrid_30X60Minute.json"
    # quadrangles_key = "CELL_NAME"
    # quad_proj = config.proj_sheets#"epsg:4267"
    synthesise_all(maps_dir, imgs_dir, quads_path)