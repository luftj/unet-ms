import os
import shutil
import random

editions_file = "E:/data/deutsches_reich/SLUB/cut/editions.txt"
images_dir = "E:/data/deutsches_reich/SLUB/cut/"

num_images_each = 2

editions_dict = {}
with open(editions_file) as fr:
    for line in fr:
        line = line.strip()
        name,edition = line.split(",")
        if not edition in editions_dict:
            editions_dict[edition] = []
        editions_dict[edition].append(name)


outdir = "data/"

for edition in editions_dict.keys():
    sampled_images = random.sample(editions_dict[edition], num_images_each)
    
    edition_dir = outdir + edition + "/"
    os.makedirs(edition_dir, exist_ok=True)
    for img in sampled_images:
        img = int(img)
        shutil.copyfile("%s/%03d.png" % (images_dir,img), "%s/%03d.png" % (edition_dir,img))
