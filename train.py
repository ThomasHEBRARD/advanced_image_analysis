import os, cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from get_datas_and_visu import (
    CreateDataset,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
)
import numpy as np
import pickle

import segmentation_models_pytorch as smp


class_rgb_values = [[0, 0, 0], [255, 255, 255]]
select_class_rgb_values = np.array(class_rgb_values)

x_train_dir = "datas/train_images"
y_train_dir = "datas/train_labels"

x_valid_dir = "datas/val_images"
y_valid_dir = "datas/val_labels"

#We define our datasets

valid_dataset = CreateDataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

train_dataset = CreateDataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


model = UNet()

TRAINING = True

# Num of epochs
EPOCHS = 70

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss functions
# Here we used two of them

loss = smp.utils.losses.CrossEntropyLoss()
#loss = smp.utils.losses.DiceLoss()

# define metrics, we use one metrics so we can compare the two training sessions
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# We only test one optimizer
optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.00008),
    ]
)

# Our moving learning rate
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1,
    T_mult=2,
    eta_min=5e-5,
)

# We can load a previous model
if os.path.exists("./models/cross_model.pth"):
    model = torch.load(
        "cross_model.pth",
        map_location=DEVICE,
    )

# Then we define our 'epochs' class with the segmentation library
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

if TRAINING:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(42, EPOCHS):

        # Perform training & validation
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        
        #We store the list of our resulsts for the training
        with open(f"./training_list_cross/train/train_logs_list_{i}.pkl", "wb") as f:
            pickle.dump(train_logs_list, f)
        with open(f"./training_list_cross/valid/valid_logs_list_{i}.pkl", "wb") as f:
            pickle.dump(valid_logs_list, f)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs["iou_score"]:
            best_iou_score = valid_logs["iou_score"]
            torch.save(model, "./models/cross_model.pth")
            print("Model saved!")
