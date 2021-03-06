import json
import os
import cv2
import numpy as np
from time import sleep
import requests
import logging
from osmtogeojson import osmtogeojson
from pyproj import Transformer

from config import path_osm, proj_map, proj_osm, proj_sheets, osm_query, force_osm_download, osm_url, draw_ocean_polygon, download_timeout

transform_osm_to_map = Transformer.from_proj(proj_osm, proj_map, skip_equivalent=True, always_xy=True)
transform_sheet_to_osm = Transformer.from_proj(proj_sheets, proj_osm, skip_equivalent=True, always_xy=True)
transform_sheet_to_map = Transformer.from_proj(proj_sheets, proj_map, skip_equivalent=True, always_xy=True)

def get_from_osm(bbox=[16.3,54.25,16.834,54.5], url = osm_url):
    os.makedirs(path_osm, exist_ok=True)
    data_path = path_osm + "rivers_%s.geojson" % "_".join(map(str,bbox))

    if proj_sheets != proj_osm: # reproject sheet bounding box to OSM coordinates
        minxy = transform_sheet_to_osm.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
        maxxy = transform_sheet_to_osm.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
        bbox = minxy+maxxy

    # prepare polygon files of ocean cover
    ocean_file_path = "%s/water_poly_%s.geojson" % (path_osm, "-".join(map(str,bbox)))
    if draw_ocean_polygon and not os.path.isfile( ocean_file_path ):
        clip_ocean_poly(bbox)
    
    # don't query if we already have the data on disk
    if not force_osm_download and os.path.isfile( data_path ):
        logging.debug("fetching osm data from disk: %s" % data_path)
        with open(data_path, encoding="utf-8") as file:
            json_data = json.load(file)
            return json_data

    sorted_bbox = ",".join(map(str,[bbox[1], bbox[0], bbox[3], bbox[2]]))
    query = osm_query.replace("{{bbox}}","%s" % sorted_bbox)
    logging.debug("osm query: %s" % query)
    # print(query)

    while True:
        try:
            result = requests.get(url, params={'data': query}, timeout=download_timeout, veri)
            result = result.json()
            break
        except Exception as e:
            import re
            error_msg = re.findall("error[^<]*",result.text)
            if len(error_msg) == 0:
                logging.critical(result.text)
                print(result.text)
                raise(e)
            logging.error(error_msg)
            if "rate_limited" in error_msg[0] or "timeout" in error_msg[0]:
                logging.warning("timeout or rate limited, retrying in 5 sec...")
                sleep(5)
                continue
            else:
                print("unknown error" + result.text)
                logging.critical("unknown error" + result.text)
                raise(e)
    gj = osmtogeojson.process_osm_json(result)
    
    with open(data_path, mode="w", encoding="utf-8") as f:
        json.dump(gj, f)

    return gj

def coord_to_point(coords, bbox, img_size, castint=True):
    if proj_osm != proj_map: # reproject osm coordinates. This gets called many times! expensive even with skip_equivalent!
        coords = transform_osm_to_map.transform(*coords)
    
    lon, lat = coords
    x = (lon-bbox[0]) / (bbox[2]-bbox[0]) * img_size[0]
    y = (lat-bbox[1]) / (bbox[3]-bbox[1]) * img_size[1]
    y = img_size[1]-y
    if castint:
        x = int(x)
        y = int(y)
    return (x,y)

def clip_ocean_poly(bbox):
    water_polys_file = "E:/data/water_polygons/simplified_water_wgs84.geojson"
    coords = " ".join(map(str,bbox))
    cropped_output_file = "%s/water_poly_%s.geojson" % (path_osm, coords.replace(" ","-"))
    print("clipping ocean...")
    command = "ogr2ogr -spat %s -clipsrc %s %s %s" % (coords, coords, cropped_output_file, water_polys_file)
    print(command)
    os.system(command)

def paint_ocean_poly(bbox):
    coords = " ".join(map(str,bbox))
    cropped_output_file = "%s/water_poly_%s.geojson" % (path_osm, coords.replace(" ","-"))
    with open(cropped_output_file) as fr:
        data = json.load(fr)
    return data["features"]

def merge_outline(geometries_):
    geometries = geometries_.copy()
    all_outlines = []
    outline = []
    idx = -1
    while len(geometries) > 0:
        old_len = len(geometries)
        idx += 1
        if idx >= len(geometries):
            idx=0

        g = geometries[idx]
        if g["type"] != "LineString": 
            geometries.remove(g)
            continue
        elif len(outline) == 0:
            outline.extend(g["coordinates"])
            geometries.remove(g)
            continue

        if g["coordinates"][0] == outline[-1]:
            outline.extend(g["coordinates"])
            geometries.remove(g)
        elif g["coordinates"][-1] == outline[0]:
            outline = g["coordinates"] + outline
            geometries.remove(g)
        else:
            for o in all_outlines:
                if g["coordinates"][0] == o[-1]:
                    o.extend(g["coordinates"])
                    geometries.remove(g)
                    break
                elif g["coordinates"][-1] == o[0]:
                    o = g["coordinates"] + o
                    geometries.remove(g)
                    break

        # print("g", g["type"], type(g["coordinates"]))
        # print(idx, old_len)
        if old_len == len(geometries):
            # no change, start new outline
            all_outlines.append(outline)
            outline = []

    # print(all_outlines)
    print(len(all_outlines))
    return all_outlines


def paint_features(json_data, bbox=[16.3333,54.25,16.8333333,54.5], img_size=(1000,850)):
    if draw_ocean_polygon:
        if proj_sheets != proj_osm: # reproject sheet bounding box to OSM coordinates
            minxy = transform_sheet_to_osm.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
            maxxy = transform_sheet_to_osm.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
            osm_bbox = minxy+maxxy
        else:
            osm_bbox = bbox
        ocean_features = paint_ocean_poly(osm_bbox)
        json_data["features"] += ocean_features

    if proj_sheets != proj_map: # reproject sheet bounding box to map coordinates
        minxy = transform_sheet_to_map.transform(bbox[0], bbox[1]) # reproject lower left bbox corner
        maxxy = transform_sheet_to_map.transform(bbox[2], bbox[3]) # reproject upper right bbox corner
        bbox = minxy+maxxy

    image = np.zeros(shape=img_size[::-1], dtype=np.uint8)
    for feature in json_data["features"]:
        try:
            # if "Lake" in feature["properties"].get("name",""):
            #     print(feature["properties"])
            if feature["geometry"]["type"] == "LineString":
                points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"] ]
                if "waterway" in feature["properties"] and feature["properties"]["waterway"] == "river":
                    thickness = 5  
                elif "natural" in feature["properties"] and feature["properties"]["natural"] == "coastline":
                    thickness = 0 if draw_ocean_polygon else 3
                else:
                    thickness = 3 # todo: move these to config
                points = np.array(points)
                cv2.polylines(image,[points],False,255,thickness=thickness)
            elif feature["geometry"]["type"] == "Polygon":
                points = [ coord_to_point(p,bbox,img_size) for p in feature["geometry"]["coordinates"][0] ]
                points = np.array(points)
                # cv2.fillConvexPoly(image, points, 255)
                if "natural" in feature["properties"] and feature["properties"]["natural"] == "coastline":
                    if draw_ocean_polygon:
                        cv2.fillPoly(image, [points], 0) # black islands on top of the white ocean
                    else:
                        cv2.polylines(image, [points], True, 255, thickness=3)
                else:
                    cv2.polylines(image, [points], True, 255, thickness=3)
                    # cv2.fillPoly(image, [points], 255)

                # draw holes
                for hole in feature["geometry"]["coordinates"][1:]:
                    points = [ coord_to_point(p,bbox,img_size) for p in hole]
                    points = np.array(points)
                    cv2.fillPoly(image, [points], 0)
                # if feature["properties"].get("name","") =="Lake Michigan":
                #     print(len(feature["geometry"]["coordinates"][0]))
                
            elif feature["geometry"]["type"] == "MultiPolygon":
                for poly in feature["geometry"]["coordinates"][0]:
                    points = [ coord_to_point(p,bbox,img_size) for p in poly ]
                    points = np.array(points)
                    # cv2.fillConvexPoly(image, points, 255)
                    cv2.polylines(image, [points], True, 255, thickness=3)
                    # cv2.fillPoly(image, [points], 255)
            elif feature["geometry"]["type"] == "GeometryCollection":
                # return None
                # print(feature["features"])
                # print(type(feature["geometry"]["geometries"]))
                # print((feature["geometry"]["geometries"][0]))
                # print(feature["properties"])

                outlines = merge_outline(feature["geometry"]["geometries"])
                # sub_features = [{
                #     "properties":feature["properties"],
                #     "geometry": {
                #         "type":"Polygon",
                #         "coordinates":[outline]
                #         }
                #     } for outline in outlines]
                # print(*[x["type"] for x in feature["geometry"]["geometries"]])
                # sub_features = [ subfeature ]#feature["geometry"]["geometries"]]
                # coll_img = np.zeros(shape=img_size[::-1], dtype=np.uint8)
                # outlines = [sum(outlines,[])]
                for outline in outlines:
                    points = [ coord_to_point(p,bbox,img_size) for p in outline ]
                    # print(points)
                    points = np.array(points)
                    # cv2.fillConvexPoly(image, points, 255)
                    cv2.polylines(image, [points], False, 255, thickness=3)
                    # cv2.fillPoly(image, [points], 255)
                # coll_img = paint_features({"features": sub_features}, bbox=bbox, img_size=img_size)
                # from matplotlib import pyplot as plt
                # plt.imshow(coll_img)
                # plt.show()
                # image = image | coll_img
                # draw holes
                for hole in [g["coordinates"][0] for g in feature["geometry"]["geometries"] if g["type"] == "Polygon"]:
                    points = [ coord_to_point(p,bbox,img_size) for p in hole]
                    points = np.array(points)
                    cv2.polylines(image, [points], False, 255, thickness=3)
                    # cv2.fillPoly(image, [points], 0)
            else:
                print(feature.get("properties",{"name":"no props"}).get("name","no name"))
                raise NotImplementedError("drawing feature type not implemented %s!" % feature["geometry"].get("type","no geomtype"))
        except Exception as e:
            logging.error(e)
            typ = feature["geometry"]["type"] if (feature is dict and "geometry" in feature and "type" in feature["geometry"]) else "no type"
            errortext = "Error parsing feature at %s with id: %s and type: %s" % (bbox, feature["properties"]["@id"], typ)
            logging.error(errortext)
    return image

if __name__ == "__main__":
    # download data
    import find_sheet
    import progressbar
    import sys
    # create necessary directories
    import os
    os.makedirs("logs/", exist_ok=True)
    path_osm = "./test_osm/"

    logging.basicConfig(filename='logs/osmkdr500.log', level=logging.DEBUG) # gimme all your loggin'!
    progress = progressbar.ProgressBar()
    sheets_file = "data/blattschnitt_dr100_regular.geojson"
    sheets_file = "E:/data/deutsches_reich/Blattschnitt/blattschnitt_dr100_merged.geojson"
    # sheets_file = "E:/data/dr500/blattschnitt_kdr500_wgs84.geojson"
    # bboxes = find_sheet.get_bboxes_from_json(sheets_file)
    bbox_dict = find_sheet.get_dict(sheets_file, True)
    bboxes = [bbox_dict["143"]]
    # bboxes = bboxes[:10]
    if len(sys.argv) == 1:
        for bbox in progress(bboxes):
            gj = get_from_osm(bbox)
    img = paint_features(gj,bbox)
    # cv2.imshow("output",img)
    # cv2.waitKey(-1)
    outpath = "test_osm/%s.png" % bbox
    cv2.imwrite(outpath, img)
    # georef this to allow easy check
    from registration import make_worldfile
    make_worldfile(outpath, bbox,[0,img.shape[0],img.shape[1],0])

