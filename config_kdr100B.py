path_osm = "E:/data/train/synthdata/kdr100B/"
path_osm = "/media/ecl2/DATA/jonas/kdr100B/osm/"

proj_map = "+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +no_defs" # Potsdam datum
proj_sheets = proj_map
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
way ({{bbox}}) [waterway=ditch];
way ({{bbox}}) [waterway=drain];
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