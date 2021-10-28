import os
from matplotlib import pyplot as plt
import ast
import score_predictions

quads_path = "E:/data/deutsches_reich/Blattschnitt/blattschnitt_dr100_regular.geojson"
raw_maps_dir = "E:/data/deutsches_reich/labelling_noise_exp/raw_maps/"
# raw_maps_dir = "E:/data/deutsches_reich/labelling_noise_exp/fix_masks/"
colour_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/colour_masks/"
test_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/manual_test/"
osm_masks_dir_A = "E:/data/deutsches_reich/labelling_noise_exp/osm_synth_A/"
osm_masks_dir_B = "E:/data/deutsches_reich/labelling_noise_exp/osm_synth_B/"
osm_masks_dir_C = "E:/data/deutsches_reich/labelling_noise_exp/osm_synth_C/"
dl_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/dl_predictions/"

# ATTENTION: can't do both A and B at once, need to change config import first
edition = "A"

if edition == "A":
    osm_masks_dir = osm_masks_dir_A
elif edition == "B":
    osm_masks_dir = osm_masks_dir_B

# synthesise dir
import config
config.water_polys_file = "E:/data/water_polygons/simplified_water_wgs84.geojson"
config.path_osm = "E:/data/train/synthdata/kdr100C/"
os.makedirs(config.path_osm, exist_ok=True)
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"
from data_handling.synthesise_data import synthesise_all
# synthesise_all(raw_maps_dir, osm_masks_dir, quads_path)

# colour thresh accuracy
# compare colour thresholding masks with manual test annotations
plot_dir = "plots/colour_manual/"
os.makedirs(plot_dir, exist_ok=True)
results = score_predictions.calc_scores_dir(colour_masks_dir, test_masks_dir, plot_dir)
with open(plot_dir + "colour_thresh_accuracy.txt", "w") as fw:
    for line in results:
        fw.write(str(line) + ",\n")
print("threshold vs manual: avg iou", sum([x["iou"] for x in results])/len(results))
print("threshold vs manual: avg ransac", sum([x["ransac"] for x in results])/len(results))
print("threshold vs manual: avg index rank", sum([x["index rank"] for x in results])/len(results))

# DL accuracy
# compare DL prediction masks with OSM annotations
plot_dir = "plots/DL_osm%s/" % edition
os.makedirs(plot_dir, exist_ok=True)
results = score_predictions.calc_scores_dir(dl_masks_dir, osm_masks_dir, plot_dir)
with open(plot_dir + "DL_osm_%s.txt" % edition, "w") as fw:
    for line in results:
        fw.write(str(line) + ",\n")
print("DL vs osm %s: avg iou" % edition, sum([x["iou"] for x in results])/len(results))
print("DL vs osm %s: avg ransac" % edition, sum([x["ransac"] for x in results])/len(results))
print("DL vs osm %s: avg index rank" % edition, sum([x["index rank"] for x in results])/len(results))

import numpy as np
results = []
with open(plot_dir + "DL_osm_%s.txt" % edition) as fr:
    for l in fr:
        d = ast.literal_eval(l[:-2])
        results.append(d)
x=[r["iou"] for r in results]
y=[r["ransac"] for r in results]
corr = np.corrcoef(x,y)[0,1]
print("correlation:", corr)
plt.scatter(x,y)
plt.xlabel("iou")
plt.ylabel("ransac")
plt.show()
