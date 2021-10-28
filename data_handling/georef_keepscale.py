import os
from PIL import Image

indirs = [
"E:/data/deutsches_reich/kdr100_6x2_editions_manualtest/A/test",
# "E:/data/deutsches_reich/kdr100_6x2_editions_manualtest/A/train",
# "E:/data/deutsches_reich/kdr100_6x2_editions_manualtest/B/test",
# "E:/data/deutsches_reich/kdr100_6x2_editions_manualtest/B/train"
]

for indir in indirs:
    for file in os.listdir(indir):
        if not os.path.splitext(file)[-1] == ".png": 
            continue
        if "_mask" in file:
            continue
        # get img size
        img = Image.open(indir+"/"+file)
        width, height = img.size
        print(indir,file, width, height)
        
        # get gcps from points file (from QGIS)
        points = []
        with open(indir+"/"+file+".points") as fr:
            fr.readline()
            for gcp_line in fr:
                easting,northing,pixel,line,_ = gcp_line.strip().split(",")
                line = line.replace("-","")
                points.append("%s %s %s %s" % (pixel, line, easting, northing))
        
        command = 'gdal_translate'
        command += ' -gcp ' + ' -gcp '.join(points) # gcps
        command += ' -a_srs "+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7" ' # set target CRS
        command += indir+"/"+file
        command += " " + os.path.splitext(indir+"/"+file)[0]+"translate.tif" # temp file name
        os.system(command)
        # os.remove(os.path.splitext(indir+"/"+file)[0]+".tif")

        command = "gdalwarp"
        command += ' -tps '
        command += " -ts %d %d " % (width, height) # target resolution
        command += os.path.splitext(indir+"/"+file)[0]+"translate.tif" # temp file name
        command += " " + os.path.splitext(indir+"/"+file)[0]+"_rpc.tif" # final file name
        os.system(command)
        os.remove(os.path.splitext(indir+"/"+file)[0]+"translate.tif")


# gdal_translate -gcp 656.48398919753174141 591.19405864197540268 12.33333333333333393 51 
# -gcp 6819.82310956790115597 603.73263888888868678 12.83333333333333393 51 
# -gcp 6821.70389660493492556 5443.93807870370255841 12.83333333333333393 50.75 
# -gcp 623.25675154321049831 5421.6820987654318742 12.33333333333333393 50.75 # line pixel easting northing
# -a_srs "+proj=longlat +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7" 
# 441.png 441_gdal.tif

# gdalwarp -ts 7539 6500  441_gdal.tif 441_gdal_warp.tif
