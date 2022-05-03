import os, cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from first_visu import (
    BuildingsDataset,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
)
import numpy as np
import pickle

import segmentation_models_pytorch as smp


class_rgb_values = [[0, 0, 0], [255, 255, 255]]
select_class_rgb_values = np.array(class_rgb_values)
DATA_DIR_VAL = "D:/DTU-Courses/DTU-AdvanceImage/mini_project/advanced_image_analysis/EM_ISBI_Challenge/EM_ISBI_Challenge/"
DATA_DIR = "D:/DTU-Courses/DTU-AdvanceImage/mini_project/advanced_image_analysis/datas/"

x_train_dir = "datas/train_images"
y_train_dir = "datas/train_labels"

x_valid_dir = "datas/val_images"
y_valid_dir = "datas/val_labels"

valid_dataset = BuildingsDataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

# Get train and val dataset instances
train_dataset = BuildingsDataset(
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

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 40

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.utils.losses.CrossEntropyLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.00008),
    ]
)

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1,
    T_mult=2,
    eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists("best_model.pth"):
    model = torch.load(
        "best_model.pth",
        map_location=DEVICE,
    )

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

    for i in range(20, EPOCHS):

        # Perform training & validation
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        with open(f"lists2/train/train_logs_list_{i}.pkl", "wb") as f:
            pickle.dump(train_logs_list, f)
        with open(f"lists2/valid/valid_logs_list_{i}.pkl", "wb") as f:
            pickle.dump(valid_logs_list, f)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs["iou_score"]:
            best_iou_score = valid_logs["iou_score"]
            torch.save(model, "./best_model.pth")
            print("Model saved!")
