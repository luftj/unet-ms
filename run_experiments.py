import logging
import os
import shutil
import random

from data_handling import synthesise_data
from data_handling.utils import is_valid_map
from data_handling import tile_images
from data_handling import random_subfiles
from data_handling import filter_tiles

# this program should run in background all the time and listens for new sheduled experiments

# initialise logger

# get experiment list from cloud
import utils.exp_schedule
infile = utils.exp_schedule.get_experiments_file("./data/")
all_exps = utils.exp_schedule.get_all_experiments(infile)

# find next scheduled experiment and check that all params are set
current_exp = utils.exp_schedule.get_next_experiment(all_exps)
print("running experiment %s" % current_exp["Exp #"])

# get params from list of experiments
# set params
map_series = current_exp["Training series"]
param_model = current_exp["model"]
param_lr = float(current_exp["learning rate"])
param_bs = int(current_exp["batch size"])
param_weight = current_exp["pos_weight"] # calculated if not given
if param_weight: param_weight = int(param_weight)
param_epochs = int(current_exp["epochs"])

# param_num_train_maps = 10 # if 0, use fixed maps, else sample randomly
# fixed_train_maps = [] # for comparable training results
# param_num_test_maps = 3 # if 0, use fixed maps, else sample randomly
fixed_test_maps = current_exp["test sheets"] # for comparability, or if manually annotated
if fixed_test_maps: fixed_test_maps = fixed_test_maps.split(",")
else: fixed_test_maps = []
param_num_test_maps = len(fixed_test_maps) if len(fixed_test_maps) > 0 else 3
fixed_train_maps = current_exp["training sheets"] # for comparability, or if manually annotated
if fixed_train_maps: fixed_train_maps = fixed_train_maps.split(",")
else: fixed_train_maps = []
param_num_train_maps = len(fixed_train_maps) if len(fixed_train_maps) > 0 else 3

param_tile_size = current_exp["tile size"]
param_size_prediction = current_exp["tile size prediction"]
param_tile_offsets = current_exp["tiling offsets"].split(",")

tile_fg_thresh = 0.01

factor_val_tiles = 0.1

def param_string():
    return "%s_%s_lr%f_bs%d" % (map_series, param_model, param_lr, param_bs)

# get data paths from config file
# todo: load from config file
path_output = "E:/experiments/deepseg_models/" + str(current_exp["Exp #"]) + "/"
path_output = "/media/ecl2/DATA/jonas/deepseg_models/" + str(current_exp["Exp #"]) + "/"
# path_output = "/media/ecl2/DATA1/jonas/deepseg_models/"
os.makedirs(path_output, exist_ok=True)
plot_dir = path_output + "/plots/" #"plots/%s/" % (current_exp["Exp #"])
os.makedirs(plot_dir, exist_ok=True)

maps_path = {
    #"USGS100": "E:/data/usgs/100k/imgs/",
    "USGS100": "/media/ecl2/DATA/jonas/usgs/100k_raw/",
    "KDR100": "E:/data/deutsches_reich/SLUB/cut/raw/",
    "kdr100_6x2_editionA_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/kdr100_6x2_editionA_manualtest/",
    "kdr100_6x2_editionB_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/kdr100_6x2_editionB_manualtest/",
    "kdr100_12x4_editionAB_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/kdr100_12x4_editionAB_manualtest/"
}
quads_path = {
    #"USGS100": "E:/data/usgs/indices/CellGrid_30X60Minute.json",
    "USGS100": "/media/ecl2/DATA/jonas/usgs/CellGrid_30X60Minute.json",
    "kdr100_6x2_editionA_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/Blattschnitt/blattschnitt_dr100_regular.geojson",
    "kdr100_6x2_editionB_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/Blattschnitt/blattschnitt_dr100_regular.geojson",
    "kdr100_12x4_editionAB_manualtest": "/media/ecl2/DATA/jonas/deutsches_reich/Blattschnitt/blattschnitt_dr100_regular.geojson"
}
index_path = {
    "USGS100": "/media/ecl2/DATA/jonas/index/",
    "kdr100_6x2_editionA_manualtest": "/media/ecl2/DATA/jonas/experiments/kdr_index_regular/",
    "kdr100_6x2_editionB_manualtest": "/media/ecl2/DATA/jonas/experiments/kdr_index_regular/",
    "kdr100_12x4_editionAB_manualtest": "/media/ecl2/DATA/jonas/experiments/kdr_index_regular/"
}
valid_map_ext = [".tif",".png"]

print("%d maps found" % len(os.listdir(maps_path[map_series])))
print("%d valid maps found" % len(list(filter(is_valid_map, os.listdir(maps_path[map_series])))))
test_path = maps_path[map_series] + "/test/"

# check if train data present
# raw_maps = list(filter(is_valid_map, os.listdir(maps_path[map_series])))
# if len(raw_maps) <= 0:
#     # if no: abort
#     raise Exception("No training maps found!") #todo: maybe download automatically from somewhere?
# todo: maybe there are no raw maps, because they are preselected

# sample a number of test maps (!= train maps)
if os.path.isdir(test_path) and len(list(filter(is_valid_map, os.listdir(test_path)))) == max(param_num_test_maps,len(fixed_test_maps)):
    print("test maps already selected")
else:
    # os.remove(test_path)
    os.makedirs(test_path) # error if exists, because we might get more test maps than we want
    if param_num_test_maps <= 0:
        # copy test files
        for test_file in fixed_test_maps:
            shutil.copyfile(maps_path[map_series]+test_file, test_path) # map
            shutil.copyfile((maps_path[map_series]+test_file).replace(".","_mask."), test_path) # mask
    else:
        # sample test maps
        fixed_test_maps = random_subfiles.sample_random(maps_path[map_series], test_path, param_num_test_maps, exclude=fixed_train_maps, nomask=True)

# sample a number of training maps (!= test maps)
train_path = maps_path[map_series] + "/train/"
if os.path.isdir(train_path) and len(list(filter(is_valid_map, os.listdir(train_path)))) == max(param_num_train_maps,len(fixed_train_maps)):
    print("train maps already selected")
else:
    os.makedirs(train_path) # error if exists, because we might get more training maps than we want
    if param_num_train_maps <= 0:
        # copy train files
        for train_file in fixed_train_maps:
            shutil.copyfile(maps_path[map_series]+train_file, train_path) # map
            shutil.copyfile((maps_path[map_series]+train_file).replace(".","_mask."), train_path) # mask
    else:
        # sample train maps
        random_subfiles.sample_random(maps_path[map_series], train_path, param_num_train_maps, fixed_test_maps, nomask=True)

# check if train data has thruth masks
synthesise_data.synthesise_maps_if_necessary(quads_path[map_series],train_path,"train")

# check if data is tiled and filtered with same settings
# print(param_tile_offsets, type(param_tile_offsets))
tiles_path = train_path + "tiles_%s_%s_%s/" % (param_tile_size, param_tile_offsets, tile_fg_thresh)
if not os.path.isdir(tiles_path): # todo: and check for files inside?
    os.makedirs(tiles_path)
    os.makedirs(tiles_path+"/imgs/")
    os.makedirs(tiles_path+"/masks/")
    # tile each map+mask
    offsets = list(map(lambda o: o.split("-"), param_tile_offsets))
    for map_file in os.listdir(train_path):
        for x_offset, y_offset in offsets:
            tile_images.tile_and_save(train_path+map_file, tiles_path, tile_size=param_tile_size, x_offset=int(x_offset), y_offset=int(y_offset), offset_step=param_size_prediction)
else:
    pass # todo: folder present, but is it correctly tiled?

# # split img and mask tiles
# map_tiles = filter(lambda f: not "_mask" in f, os.listdir(tiles_path))
# for map_tile in map_tiles:
#     mask_tile = map_tile.replace(".","_mask.")
#     shutil.move(tiles_path+map_tile, tiles_path+"/imgs/"+map_tile) # imgs
#     shutil.move(tiles_path+mask_tile, tiles_path+"/masks/"+map_tile) # masks

# filter tiles
fg_bg_factor = filter_tiles.filter_dir(tiles_path, tiles_path, tile_fg_thresh)
map_tiles = list(filter(is_valid_map, os.listdir(tiles_path+"/imgs/")))
current_exp["num training tiles"] = len(map_tiles)
print("%d train tiles after filtering" % len(map_tiles))
print("fg/bg factor: %f" % (fg_bg_factor))
if not param_weight:
    param_weight = 1.0/fg_bg_factor
    current_exp["pos_weight"] = param_weight

# random train-val split 
val_tiles_path = maps_path[map_series] + "/val/tiles_%s_%s_%s/" % (param_tile_size, param_tile_offsets, tile_fg_thresh)
if not os.path.isdir(val_tiles_path):
    os.makedirs(val_tiles_path)  # error if exists, because we might get more validation tiles than we want
    os.makedirs(val_tiles_path+"/imgs/")
    os.makedirs(val_tiles_path+"/masks/")
    num_val_maps = int(factor_val_tiles*len(map_tiles))
    val_map_tiles = random.sample(map_tiles,num_val_maps)
    for map_tile in val_map_tiles:
        shutil.move(tiles_path+"/imgs/"+map_tile, val_tiles_path+"/imgs/"+map_tile)
        shutil.move(tiles_path+"/masks/"+map_tile, val_tiles_path+"/masks/"+map_tile)
else:
    print("%d val tiles already sampled" % len(os.listdir(val_tiles_path+"/imgs/")))

# check if already trained
path_model = path_output + "model_%s.net" % (param_string())
train_logfile = os.path.splitext(path_model)[0]+"_log.txt"
if not os.path.isfile(path_model):
    # if no: train new model with params
    print("training new model at %s..." % path_model)
    # import chosen model implementation
    if param_model == "persson":
        pass
    elif param_model == "persson_32-64-128-256":
        pass
    elif param_model == "eth":
        pass
        from persson_unet import train_eth
        train_eth.LEARNING_RATE = param_lr
        train_eth.BATCH_SIZE = param_bs
        train_eth.pos_weight = param_weight
        train_eth.NUM_EPOCHS = param_epochs
        train_eth.TRAIN_IMG_DIR = tiles_path+"/imgs/"
        train_eth.TRAIN_MASK_DIR = tiles_path+"/masks/"
        train_eth.VAL_IMG_DIR = val_tiles_path+"/imgs/"
        train_eth.VAL_MASK_DIR = val_tiles_path+"/masks/"
        train_eth.logfile = train_logfile
        os.makedirs("saved_images/pred_tiles/", exist_ok=True) # todo: this should be changed in training implementation
        # run training 
        train_eth.main()
    else:
        raise NotImplementedError("can't find model implementation: %s" % param_model)
    # todo: set data augmentation
    # todo: keep track of dice score and avg loss over the epochs

    from utils import train_scoring
    # get training scores
    # todo: add these to experiments spreadsheet
    current_exp["lowest loss"] = "%s@%s" % (train_scoring.get_best_loss(logfile=train_logfile))
    train_dice,best_epoch = train_scoring.get_best_dice(logfile=train_logfile)
    current_exp["highest train dice"] = "%s@%s" % (train_dice,best_epoch)
    # make a plot of training run: e.g. loss over epochs, mark selected model
    train_scoring.plot_train_scores(logfile=train_logfile, outdir=plot_dir)

    # select epoch with best dice score for prediction
    print("best model at epoch: %d" % best_epoch)
    # save best model to model path
    shutil.move("checkpoint_%d.pth.tar" % best_epoch, path_model)
    # todo: remove all other model files

# # check if test data is present
#     # if no: abort

# check if test data has truth masks
synthesise_data.synthesise_maps_if_necessary(quads_path[map_series],test_path,"test")

# check if test data is tiled with same settings
tiles_path = test_path + "tiles_%s_%s_%s/" % (param_tile_size, param_tile_offsets, tile_fg_thresh)
if not os.path.isdir(tiles_path): # todo: and check for files inside?
    # if no: tile test data
    os.makedirs(tiles_path)
    os.makedirs(tiles_path+"/imgs/")
    os.makedirs(tiles_path+"/masks/")
    # tile each map+mask
    offsets = list(map(lambda o: o.split("-"), param_tile_offsets))
    for map_file in os.listdir(test_path):
        for x_offset, y_offset in offsets:
            tile_images.tile_and_save(test_path+map_file, tiles_path, tile_size=param_tile_size, x_offset=int(x_offset), y_offset=int(y_offset), offset_step=200)
else:
    print("test tiles present") # todo: folder present, but is it correctly tiled?

# run predictions on test tiles
import persson_unet.predict_eth
persson_unet.predict_eth.VAL_IMG_DIR = tiles_path+"/imgs/"
persson_unet.predict_eth.VAL_MASK_DIR = tiles_path+"/masks/"
persson_unet.predict_eth.model_path = path_model #load model from best epoch
test_dice = persson_unet.predict_eth.main() # returns test score

# merge test tiles to full predictions
import merge_tiles
os.makedirs(path_output + "/test/", exist_ok=True)
merge_tiles.merge_dir("predictions/pred_tiles/", path_output + "/test/")
print("saved test predictions to %s" % (path_output + "/test/"))
shutil.rmtree("predictions/")

# if not index present
# if not os.path.isfile(index_path[map_series]+"/index.ann"):
#     os.makedirs(index_path[map_series], exist_ok=True)
#     #   build index
#     print("rebuilding index...")
#     raise NotImplementedError("index rebuilding not imported yet...")
#     # create folders first
#     os.makedirs(config.reference_descriptors_folder, exist_ok=True)
#     os.makedirs(config.reference_keypoints_folder, exist_ok=True)
#     import indexing
#     indexing.build_index(quads_path[map_series])

# calculate scores for each prediction
current_exp["test dice"] = test_dice
# todo: index rank
import score_predictions
pred_results = score_predictions.calc_scores_dir(path_output + "/test/", test_path, plot_dir) # todo: get index rank as well
current_exp["avg test iou"] = sum([x["iou"] for x in pred_results])/len(pred_results)
current_exp["avg test dice"] = sum([x["dice"] for x in pred_results])/len(pred_results)
current_exp["avg test index rank"] = sum([x["index rank"] for x in pred_results])/len(pred_results)
current_exp["avg test num matches"] = sum([x["num matches"] for x in pred_results])/len(pred_results)
current_exp["avg test ransac"] = sum([x["ransac"] for x in pred_results])/len(pred_results)
# todo: also calculate accuracy, precision, recall, ...

# append scores to list of experiment
utils.exp_schedule.write_experiment_to_excel(infile, current_exp)
utils.exp_schedule.upload_experiments_file(infile, "phd/deep_segmentation_experiments.xlsx")
print("uploaded results for exp %s" % current_exp)

# find experiment with best result so far?
# make some plots for the experiments
# plot avg score comparison between different experiments
# how to choose sensible experiments from table?
# maybe manual list and compare the current to manually set "best so far"

# compare architectures: make plot of best experiment for each architecture -> best according to which score?
# compare map series