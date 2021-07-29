path_osm = "E:/data/train/synthdata/usgs100/"

proj_map = "EPSG:4267"#(NAD27)
proj_sheets = "EPSG:4267"#(NAD27) #proj_map
proj_osm = "EPSG:4326"#(WGS84)
#"EPSG:4269"#(NAD83) 
#"EPSG:4267"#(NAD27)

# osm_url = "https://nc.hcu-hamburg.de/overpass_us/api/interpreter"
osm_url = "http://overpass-api.de/api/interpreter/"
osm_query = """[out:json];
                (
                nwr ({{bbox}}) [water=lake]; 
                nwr ({{bbox}}) [water=reservoir]; 
                nwr ({{bbox}}) [natural=water] [name]; 
                way ({{bbox}}) [type=waterway] [name]; 
                way ({{bbox}}) [waterway=river] [name];
                way ({{bbox}}) [waterway=canal] [name];
                way ({{bbox}}) [water=river];
                way ({{bbox}}) [waterway=stream] [name];
                way ({{bbox}}) [waterway=riverbank];
                way ({{bbox}}) [natural=coastline];
                );
                out body;
                >;
                out skel qt;"""
force_osm_download = True
download_timeout = (5,600) # connect timeout, read timeout
draw_ocean_polygon = False
quadrangles_key = "CELL_NAME"