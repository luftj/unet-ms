import tkinter as tk
from PIL import Image, ImageTk
import argparse
import os
from functools import partial

images_list = []
cur_img_idx = -1

labels = []

category_labels = ["blank", "margin", "background", "linear black (streams)", "linear coloured (rivers)", "area (lakes, ocean, big rivers)", "speckles (ponds)"]

def next_image_cb(e):
    next_image()

def next_image():
    # store old state
    global cur_img_idx
    if cur_img_idx >= 0:
        state = [str(int(b["bg"] == "green")) for b in but_labels]
        labels.append(state)
    
    if not "1" in state:
        # no categories set, this can't be right
        print("please set at least one category!")
    else:
        # increment current image
        cur_img_idx += 1

    # update status bar at bottom
    label_progress["text"] = "%d/%d" % (cur_img_idx, len(images_list))

    if cur_img_idx >= len(images_list):
        print("no more images")
        finish(None)

    # load map image
    current_image_path = images_list[cur_img_idx]
    main_win.title("sorting: %s" % current_image_path)

    img = Image.open(imgs_path + current_image_path) 
    mask = Image.open(masks_path + current_image_path)
    
    # show images
    photo = ImageTk.PhotoImage(img)
    label_img["image"] = photo
    label_img.configure(image=photo)
    label_img.image = photo
    photo = ImageTk.PhotoImage(mask)
    label_mask["image"] = photo
    label_mask.configure(image=photo)
    label_mask.image = photo

    # reset buttons
    for b in but_labels:
        b["bg"] = "grey"

def set_category_cb(i,e):
    set_category(i)
def set_category(i):
    # colour button to indicate toggle state
    if but_labels[i]["bg"] == "green":
        but_labels[i]["bg"] = "gray"
    else:
        but_labels[i]["bg"] = "green"

def finish_cb(e):
    finish()

def finish():
    # save results to csv
    with open(args.output, "w") as fw:
        fw.write('"img","%s"\n'%('","'.join(category_labels)))
        for idx,state in enumerate(labels):
            if not "1" in state:
                # don't store img without category
                continue
            fw.write("%s,%s\n" % (images_list[idx],",".join(state)))

    main_win.destroy()
    exit()

def check_continue():
    if not os.path.exists(args.output):
        return
    # else: csv found, continue
    with open(args.output) as fr:
        headers = fr.readline().strip().split('","')
        global category_labels
        category_labels = [x.replace("\"","") for x in headers[1:]]
        # print(category_labels)
        
        content = fr.readlines()
        global cur_img_idx
        cur_img_idx = len(content) - 1

        global labels
        labels = [line.strip().split(",")[1:] for line in content]
        # print(*labels, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('output', help='output directory')
    args = parser.parse_args()
    # python tile_sorter.py /e/data/train/AB_tiles/ /e/data/train/AB_tiles_sorted.csv

    imgs_path = args.input + "/imgs/"
    masks_path = args.input + "/masks/"
    images_list = os.listdir(imgs_path)

    # allow to continue from given csv
    check_continue()

    main_win = tk.Tk()
    main_win.title("sorting: %s" % "...")

    label_img = tk.Label(main_win)
    label_img.pack(side="left")
    label_mask = tk.Label(main_win)
    label_mask.pack(side="right")

    but_next = tk.Button(main_win, text="Next [RETURN]", command=next_image)
    but_next.pack()
    but_labels = []
    for idx,cat in enumerate(category_labels):
        b = tk.Button(main_win, text="%s [%s]" %(cat, idx), command=partial(set_category,idx))
        b.pack()
        but_labels.append(b)
        main_win.bind("%s" %idx, partial(set_category_cb,idx))
    but_quit = tk.Button(main_win, text="Save & Quit [ESCAPE]", command=finish)
    but_quit.pack()
    label_progress = tk.Label(main_win,text="0/0")
    label_progress.pack()
    main_win.bind("<Return>", next_image_cb)
    main_win.bind("<Escape>", finish_cb)
    next_image() # start with first image immediately
    main_win.mainloop()