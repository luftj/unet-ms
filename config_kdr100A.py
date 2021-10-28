path_osm = "E:/data/train/synthdata/kdr100A/"
path_osm = "/media/ecl2/DATA/jonas/kdr100A/osm/"

proj_map = "EPSG:4314"#"+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +no_defs" # Potsdam datum
proj_sheets = "EPSG:4314"#proj_map
proj_osm = "EPSG:4326"

valid_map_ext = [".tif",".png"]

osm_url = "https://nc.hcu-hamburg.de/overpass_europe/api/interpreter" # no trailing slash!
# osm_url = "http://overpass-api.de/api/interpreter"
osm_query = """[out:json];
(
nwr ({{bbox}}) [water=lake]; 
nwr ({{bbox}}) [water=reservoir]; 
way ({{bbox}}) [natural=water] [name]; 
way ({{bbox}}) [type=waterway] [name]; 
way ({{bbox}}) [waterway=river] [name];
way ({{bbox}}) [waterway=canal] [name];
way ({{bbox}}) [water=river];
way ({{bbox}}) [waterway=stream] [name];
way ({{bbox}}) [natural=coastline];
way ({{bbox}}) [waterway=riverbank];
);
out body;
>;
out skel qt;"""
force_osm_download = True
download_timeout = (5,600) # connect timeout, read timeout
draw_ocean_polygon = True
fill_polys = True
quadrangles_key = "blatt_100"
sheet_name_field = "blatt_100"

# indexing
base_path_index = "E:/experiments/kdr_index_regular/"
base_path_index = "/media/ecl2/DATA/jonas/experiments/kdr_index_regular/"
reference_sheets_path = base_path_index+"index/sheets.clf"
reference_index_path = base_path_index+"index/index.ann"
reference_descriptors_path = base_path_index+"index/index.clf"
reference_descriptors_folder = base_path_index+"index/descriptors"
reference_keypoints_path = base_path_index+"index/keypoints.clf"
reference_keypoints_folder = base_path_index+"index/keypoints"

from cv2 import INTER_AREA, INTER_LINEAR, INTER_CUBIC
resizing_index_query = INTER_AREA
resizing_register_query = INTER_AREA
resizing_register_reference = INTER_CUBIC

index_img_width_query = 500
index_n_descriptors_query = 500
index_k_nearest_neighbours = 50
index_voting_scheme = "antiprop"
index_lowes_test_ratio = None # 0.8

# the following indexing parameters require rebuilding the index
index_img_width_train = 500
index_border_train = 30
index_annoydist = "euclidean"
index_n_descriptors_train = 300
detector = kp_detector = "kaze_upright"
# possible detectors: "kaze_upright","akaze_upright","sift","surf_upright","ski_fast","cv_fast"
index_descriptor_length = 64 # depends on detector!
index_num_trees = 10