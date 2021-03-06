from albumentations.augmentations.geometric.rotate import RandomRotate90
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from model_eth import UNET
from utils_eth import (
    load_checkpoint,
    get_val_loader,
    check_accuracy,
    save_predictions_as_imgs,
)
import os

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 3
NUM_WORKERS = 2
pos_weight = 60
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def main():
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Crop(x_min=60,x_max=260,y_min=60,y_max=260,p=1),# only crop mask, not img
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(torch.cuda.FloatTensor([pos_weight]) if DEVICE=="cuda" else torch.FloatTensor([pos_weight])))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    val_loader = get_val_loader(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)

    # print predictions to a folder
    os.makedirs("predictions/", exist_ok=True)
    save_predictions_as_imgs(
        val_loader, model, folder="predictions/", device=DEVICE
    )


if __name__ == "__main__":
    main()