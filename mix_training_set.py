import tkinter as tk
from PIL import Image
import numpy as np
import argparse
import os
import shutil
from functools import partial
import random
from PIL import Image

selected_images = {}

def resample(category_numbers):
    for c,n in category_numbers.items():
        print(c,n)
        if n > len(categories[c]):
            # add random duplicate images for data augmentation
            selected_images[c] = categories[c]*(n//len(categories[c])) + random.sample(categories[c],n%len(categories[c]))
        else:
            # sample a subset of the available images
            selected_images[c] = random.sample(categories[c],n)
    update_info()

def update_info():
    # show distributions
    print("class incidence:", ["%s: %d" % (k, len(v)) for k,v in selected_images.items()])
    print("class fg%:", ["%s: %0.3f" % (k, sum(list(zip(*[imgs_class_pixels[i] for i in v]))[0])/sum(list(zip(*[imgs_class_pixels[i] for i in v]))[1]) ) for k,v in selected_images.items() if len(v)>0])
    print()
    # fg, bg = list(zip(*imgs_class_pixels.values())) # doesn't respect selection
    total_pxs = sum([[imgs_class_pixels[i] for i in c] for c in selected_images.values()],[])
    # print(l)
    # map(total_pxs.extend,l)
    # print(total_pxs)
    fg,bg = list(zip(*total_pxs))
    print( "fg/bg total: %0.2f" % (sum(fg)/sum(bg)))
    label_fg["text"] = "fg/bg: %0.3f%%\n#imgs: %d" % (sum(fg)/sum(bg),len(sum(selected_images.values(),[])))

def slider_update(category,val):
    print(category,val)
    category_numbers = {c:len(v) for c,v in selected_images.items()}
    category_numbers[category] = int(val)
    resample(category_numbers)

def singlify_class_label(category_names,labels):
    label_precedence = ["blank", "linear black (streams)", "linear coloured (rivers)", "area (lakes, ocean, big rivers)", "speckles (ponds)",  "margin", "background"]
    most_relevant = len(label_precedence)
    global categories
    for idx,l in enumerate(labels):
        if l == "1":
            cat = list(category_names)[idx]
            precedence = label_precedence.index(cat)
            if precedence < most_relevant:
                most_relevant = precedence
    return label_precedence[most_relevant]

def get_images_properties(masks_dir, csv_path):
    # load csv
    with open(csv_path) as fr:
        headers = fr.readline().strip().split('","')
        categories = {x.replace("\"",""):[] for x in headers[1:]}
        print(categories)
        imgs_class_pixels = {}
        
        # for each image listed in csv
        for line in fr:
            file, *labels = line.strip().split(",")
            # load image
            img = Image.open(masks_dir+"/"+file)
            pixels = np.array(img)
            # calculate number of class pixels
            num_bg = pixels.size # todo: allow more than two classes
            num_fg = np.count_nonzero(pixels)
            imgs_class_pixels[file] = (num_fg,num_bg)
            # # sum up image classes
            # for idx,l in enumerate(labels):
            #     if l == "1":
            #         categories[list(categories.keys())[idx]].append(file)

            # put image in category bin
            categories[singlify_class_label(categories.keys(), labels)].append(file)
        # store all props in var
        # print(*categories.items(),sep="\n")
        # print(imgs_class_pixels)
    return categories, imgs_class_pixels

def load_and_augment(img_path, aug_mode):
    img = Image.open(img_path)
    if aug_mode == "dup":
        pass
    elif aug_mode == "flip_hor":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_mode == "flip_ver":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise NotImplemented("augmentation mode '%s' not implemented" % aug_mode)
    return img

def compile_dataset(in_dir, out_dir):
    os.makedirs(out_dir+"/masks/", exist_ok=True)
    os.makedirs(out_dir+"/imgs/", exist_ok=True)
    
    all_selected_images = sum(selected_images.values(),[])
    duplicates = {x:all_selected_images.count(x) for x in all_selected_images}

    augmentation_modes = ["dup","flip_hor", "flip_ver"]

    for image, count in duplicates.items():
        print("saving %s" % image)
        # copy image
        shutil.copyfile("%s/imgs/%s" % (in_dir,image), "%s/imgs/%s" % (out_dir,image))
        # copy mask
        shutil.copyfile("%s/masks/%s" % (in_dir,image), "%s/masks/%s" % (out_dir,image))
        for i in range(1,count):
            # choose augmentation randomly
            aug_mode = random.choice(augmentation_modes)
            # augment image
            img = load_and_augment("%s/imgs/%s" % (in_dir,image),aug_mode)
            mask = load_and_augment("%s/masks/%s" % (in_dir,image),aug_mode)
            # save augmented image
            img.save("%s/imgs/%s_%d.%s" % (out_dir,image.split(".")[0],i,image.split(".")[1]))
            mask.save("%s/masks/%s_%d.%s" % (out_dir,image.split(".")[0],i,image.split(".")[1]))
    print("done compiling dataset!")
    print("distribution:",["%s: %d" % (k, len(v)) for k,v in selected_images.items()])
    main_win.destroy()
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', help='input directory')
    parser.add_argument('labels', help='input class labels')
    parser.add_argument('output', help='output directory')
    args = parser.parse_args()
    # python mix_training_set.py /e/data/train/AB_tiles/ /e/data/train/AB_tiles_sorted.csv /e/data/train/AB_tiles_remix/
    
    # imgs_path = args.indir + "/imgs/"
    masks_path = args.indir + "/masks/"
    # images_list = os.listdir(imgs_path)

    categories, imgs_class_pixels = get_images_properties(masks_path, args.labels)
    selected_images = categories.copy()
    
    main_win = tk.Tk()
    main_win.title("Mixing dataset from %s" % args.indir)

    label_fg = tk.Label(main_win, text="0")
    label_fg.grid(row=0,column=0)
    button_compile = tk.Button(main_win, text="Compile dataset", command=partial(compile_dataset,args.indir,args.output))
    button_compile.grid(row=1,column=0)

    max_augmentation_factor = 4
    
    for idx, (category, imgs) in enumerate(categories.items()):
        w = tk.Scale(main_win, to=0, from_=max_augmentation_factor*len(imgs), label=len(imgs), command=partial(slider_update,category))
        w.grid(row=0, column=idx+1)
        w.set(len(imgs))
        wl = tk.Label(main_win, text=category).grid(row=1, column=idx+1)


    main_win.mainloop()