import os, cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import random, tqdm
import matplotlib.pyplot as plt
import albumentations as album

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

x_train_dir = "datas/train_images"
y_train_dir = "datas/train_labels"

x_valid_dir = "datas/val_images"
y_valid_dir = "datas/val_labels"


# Get the visualization
def visualize(number,**images):
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace("_", " ").title(), fontsize=20)
        plt.imshow(image)
    plt.savefig('predictions/result_visualization_{}.png'.format(number))
    plt.show()
    


# We one encode our labels to make it works with our loss function
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# We reverse the one hot to get the results
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


# We encode in color in order to plot the image
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

#We create our pytorch dataset using the previous functions
class CreateDataset(torch.utils.data.Dataset):


    def __init__(
        self,
        images_dir,
        masks_dir,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.image_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.mask_paths = [
            os.path.join(masks_dir, image_id)
            for image_id in sorted(os.listdir(masks_dir))
        ]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks with cv2
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)

#Here we define our training augmentation
def get_training_augmentation():
    #With a size of 512 for our initial images we crop half of the image
    #We then apply random flip and rotation
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(
            min_height=1536, min_width=1536, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

#We define a prepocessing function in order to work with the albumentation library
def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


class_rgb_values = [[0, 0, 0], [255, 255, 255]]
select_class_rgb_values = np.array(class_rgb_values)

