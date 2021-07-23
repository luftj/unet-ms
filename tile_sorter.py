import tkinter as tk
from PIL import Image, ImageTk
import argparse
import os
from functools import partial

images_list = []
cur_img_idx = -1

labels = []

category_labels = ["blank", "margin", "background", "linear black (streams)", "linear coloured (rivers)", "area (lakes, ocean, big rivers)", "speckles (ponds)"]

def back_image(e=None):
    next_image(reverse=True)

def next_image(e=None, reverse=False):
    global cur_img_idx
    # store old state
    if cur_img_idx >= 0:
        state = [str(int(b["relief"] == "sunken")) for b in but_labels]
        if len(labels) > cur_img_idx:
            labels[cur_img_idx] = state
        else:
            labels.append(state)
    
    if reverse:
        if cur_img_idx > 0:
            cur_img_idx -= 1 # go back one image
        else:
            return # don't reverse before start
    else:    
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
        finish() # save and exit

    # load map images
    current_image = images_list[cur_img_idx]
    main_win.title("Current: %s" % current_image)

    img = Image.open(imgs_path + current_image) 
    mask = Image.open(masks_path + current_image)
    
    # show images
    photo_img = ImageTk.PhotoImage(img)
    label_img["image"] = photo_img
    label_img.configure(image=photo_img)
    label_img.image = photo_img
    photo_mask = ImageTk.PhotoImage(mask)
    label_mask["image"] = photo_mask
    label_mask.configure(image=photo_mask)
    label_mask.image = photo_mask

    # reset buttons
    for idx,b in enumerate(but_labels):
        if len(labels) > cur_img_idx:
            # we have seen this tile before, load state
            b["relief"] = "sunken" if labels[cur_img_idx][idx]=="1" else "raised"
        else: # default state
            b["relief"] = "raised"

def set_category(i, e=None):
    # colour button to indicate toggle state
    if but_labels[i]["relief"] == "sunken":
        but_labels[i]["relief"] = "raised"
    else:
        but_labels[i]["relief"] = "sunken"

def finish(e=None):
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
        
        # read all previous labels
        content = fr.readlines()
        global cur_img_idx
        cur_img_idx = len(content)
        print("start idx", cur_img_idx)
        print("continuing from", content[-1])

        global labels
        labels = [line.strip().split(",")[1:] for line in content]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input directory')
    parser.add_argument('output', help='output csv file')
    args = parser.parse_args()
    # python tile_sorter.py /e/data/train/AB_tiles/ /e/data/train/AB_tiles_sorted.csv

    imgs_path = args.input + "/imgs/"
    masks_path = args.input + "/masks/"
    images_list = os.listdir(imgs_path)

    # load and continue from output csv
    check_continue()

    main_win = tk.Tk()
    main_win.title("Current: %s" % "...")

    label_img = tk.Label(main_win)
    label_img.pack(side="left")
    label_mask = tk.Label(main_win)
    label_mask.pack(side="right")

    but_next = tk.Button(main_win, text="Next [RETURN]", command=next_image)
    but_next.pack()
    main_win.bind("<Return>", next_image)
    but_back = tk.Button(main_win, text="Back [BACKSPACE]", command=back_image)
    but_back.pack()
    main_win.bind("<BackSpace>", back_image)
    but_labels = []

    # category buttons
    for idx,cat in enumerate(category_labels):
        b = tk.Button(main_win, text="%s [%s]" %(cat, idx), command=partial(set_category,idx))
        b.pack()
        but_labels.append(b)
        main_win.bind("%s" %idx, partial(set_category,idx))
    
    but_quit = tk.Button(main_win, text="Save & Quit [ESCAPE]", command=finish)
    but_quit.pack()
    main_win.bind("<Escape>", finish)

    # status
    label_progress = tk.Label(main_win, text="0/0")
    label_progress.pack()

    next_image() # start with first image immediately
    main_win.mainloop()