import os
from matplotlib import pyplot as plt
import ast
import score_predictions

quads_path = "E:/data/deutsches_reich/Blattschnitt/blattschnitt_dr100_regular.geojson"
raw_maps_dir = "E:/data/deutsches_reich/labelling_noise_exp/raw_maps/"
colour_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/colour_masks/"
dl_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/dl_predictions/"
test_masks_dir = "E:/data/deutsches_reich/labelling_noise_exp/manual_test/"

# colour thresh accuracy
# compare colour thresholding masks with manual test annotations
# plot_dir = "plots/colour_manual/"
# os.makedirs(plot_dir, exist_ok=True)
# results = score_predictions.calc_scores_dir(colour_masks_dir, test_masks_dir, plot_dir)
# with open(plot_dir + "colour_thresh_accuracy.txt", "w") as fw:
#     for line in results:
#         fw.write(str(line) + ",\n")
# print("colour vs manual: avg iou", sum([x["iou"] for x in results])/len(results))
# print("colour vs manual: avg ransac", sum([x["ransac"] for x in results])/len(results))
# print("colour vs manual: avg index rank", sum([x["index rank"] for x in results])/len(results))

# compare DL predictions with manual test
plot_dir = "plots/dl_manual/"
os.makedirs(plot_dir, exist_ok=True)
results = score_predictions.calc_scores_dir(dl_masks_dir, test_masks_dir, plot_dir)
with open(plot_dir + "dl_seg_accuracy.txt", "w") as fw:
    for line in results:
        fw.write(str(line) + ",\n")
print("DL vs manual: avg iou", sum([x["iou"] for x in results])/len(results))
print("DL vs manual: avg ransac", sum([x["ransac"] for x in results])/len(results))
print("DL vs manual: avg index rank", sum([x["index rank"] for x in results])/len(results))

# compare colour thresholding with DL predictions
plot_dir = "plots/colour_dl/"
os.makedirs(plot_dir, exist_ok=True)
results = score_predictions.calc_scores_dir(colour_masks_dir, dl_masks_dir, plot_dir)
with open(plot_dir + "colour_dl_compare.txt", "w") as fw:
    for line in results:
        fw.write(str(line) + ",\n")
print("colour vs DL: avg iou", sum([x["iou"] for x in results])/len(results))
print("colour vs DL: avg ransac", sum([x["ransac"] for x in results])/len(results))
print("colour vs DL: avg index rank", sum([x["index rank"] for x in results])/len(results))

# import numpy as np
# results = []
# with open(plot_dir + "colour_thresh_accuracy.txt") as fr:
#     for l in fr:
#         d = ast.literal_eval(l[:-2])
#         results.append(d)
# x=[r["iou"] for r in results]
# y=[r["ransac"] for r in results]
# corr = np.corrcoef(x,y)[0,1]
# print("correlation:", corr)
# plt.scatter(x,y)
# plt.xlabel("iou")
# plt.ylabel("ransac")
# plt.show()
