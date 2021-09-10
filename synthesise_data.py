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

def get_coords_from_raster(georef_image_path, point):
    import rasterio # leave this in here, to avoid proj_db errors?
    dataset = rasterio.open(georef_image_path)

    latlong = dataset.transform * point

    return latlong

def get_pixel_from_raster(georef_image_path, lonlat):
    import rasterio # leave this in here, to avoid proj_db errors?

    dataset = rasterio.open(georef_image_path)
    tr = Transformer.from_proj(config.proj_sheets, dataset.crs, skip_equivalent=True, always_xy=True)
    lonlat = tr.transform(*lonlat)
    xy = dataset.index(*lonlat)[::-1]

    return xy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('quads', help='quadrangles geojson file')
    args = parser.parse_args()

    maps_dir = args.input#"E:/data/usgs/100k"
    imgs_dir = args.outdir#+"/imgs/"#"E:/data/usgs/100k/imgs"
    # masks_dir = args.outdir+"/masks/"#"E:/data/usgs/100k/masks"
    # quadrangles_file = "E:/data/usgs/indices/CellGrid_30X60Minute.json"
    # quadrangles_key = "CELL_NAME"
    # quad_proj = config.proj_sheets#"epsg:4267"

    # load quadrangles data
    with open(args.quads, encoding="utf8") as fr:
        quadrangles = { f["properties"][config.quadrangles_key] : f["geometry"]["coordinates"][0] for f in json.load(fr)["features"]}

    # iterate over maps
    valid_ext = [".tif"]
    for file in os.listdir(maps_dir): # ["PA_Bradford_170431_1980_100000_geo.tif"]:#
        if not os.path.splitext(file)[-1] in valid_ext:
            continue
        map_name = file.split("_")[1]
        map_name = map_name.replace("St ","Saint ")
        print(file)
        print(map_name)
        # ret = osm.proj_map = subprocess.run("gdalinfo "+ maps_dir+"/"+file)
        result = subprocess.check_output("gdalinfo \""+ maps_dir+"/"+file+"\"", shell=True)
        # print(result.decode("ascii"))
        # print()
        crs = re.findall(r"(?<=Coordinate System is:).*(?=Data axis|Origin)",result.decode("ascii"), flags=re.MULTILINE+re.DOTALL)
        # print(crs[0].replace("\r","").replace("\n",""))
        osm.proj_map = crs[0].replace("\r","").replace("\n","")
        osm.transform_osm_to_map = Transformer.from_proj(osm.proj_osm, osm.proj_map, skip_equivalent=True, always_xy=True)
        osm.transform_sheet_to_map = Transformer.from_proj(osm.proj_sheets, osm.proj_map, skip_equivalent=True, always_xy=True)
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
        print("quad",quad)

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
        margin_right = xs[-1]#max(xs)
        margin_top = min(ys)
        margin_bottom = max(ys)
        print(margin_left,    margin_right,    margin_top,    margin_bottom   )
        # download osm data
        xs, ys = list(zip(*quad))
        bbox = [min(xs),min(ys),max(xs),max(ys)]
        osm_features = osm.get_from_osm(bbox)
        img_size = (map_size[0]-margin_left-(map_size[0]-margin_right),map_size[1]-margin_top-(map_size[1]-margin_bottom))
        print(map_size, img_size)
        img = osm.paint_features(osm_features,bbox, img_size=img_size)
        if img is None:
            print("error in painting OSM mask, skipping")
            continue
        
        # pad OSM with margin
        full_img=Image.new("L", map_size, 0)
        print("left neatline:",(margin_left))
        print("right neatline:",(img_size[0]+margin_left))
        full_img.paste(Image.fromarray(img), (margin_left,margin_top))

        # from matplotlib import pyplot as plt
        # plt.imshow(full_img)
        # plt.show()

        # save osm mask
        # os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)
        full_img.save(imgs_dir + "/" + os.path.splitext(file)[0]+"_mask.tif")
        # shutil.copy(maps_dir + "/" + file, imgs_dir + "/" + file)
        
        # tile images after
