from albumentations.augmentations.geometric.rotate import RandomRotate90
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from persson_unet.model_eth import UNET
from persson_unet.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = 2 if DEVICE=="cuda" else 0
pos_weight = 60
IMAGE_HEIGHT = 320  # 1280 originally
IMAGE_WIDTH = 320  # 1918 originally
maskcrop = 60 # masks are 200x200
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
logfile = "train_log.txt"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    avg_loss = 0
    for batch_idx, (data, targets, names) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        avg_loss += loss.item()
    
    avg_loss /= batch_idx
    return avg_loss

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            A.RandomRotate90(p=0.1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        maskcrop=maskcrop
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    with open(logfile, "w") as log:
        log.write("epoch,score,loss\n")
        for epoch in range(NUM_EPOCHS):
            print("Epoch: %s" % epoch)
            avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint,filename="checkpoints/checkpoint_%d.pth.tar" % epoch)

            # check accuracy
            score = check_accuracy(val_loader, model, device=DEVICE)
            log.write("%d,%f,%f\n" % (epoch, score, avg_loss))

            # print some examples to a folder
            # save_predictions_as_imgs(
            #     val_loader, model, folder="saved_images/pred_tiles/", device=DEVICE, maskcrop=maskcrop
            # )


if __name__ == "__main__":
    main()
