# importing the libraries
import numpy as np

# for reading and displaying images
from skimage.io import imread
import torch


def get_datas(nb_picture):
    # loading training images
    train_img = []
    for nb_image in range(1, nb_picture + 1):
        # defining the image path
        if nb_image < 10:
            image_path = (
                "EM_ISBI_Challenge/EM_ISBI_Challenge/train_images/train_0{}.png".format(
                    nb_image
                )
            )
        if nb_image >= 10:
            image_path = (
                "EM_ISBI_Challenge/EM_ISBI_Challenge/train_images/train_{}.png".format(
                    nb_image
                )
            )
        # reading the image
        img = imread(image_path, as_gray=True)
        # normalizing the pixel values
        img = img / 255.0
        # converting the type of pixel to float 32
        img = img.astype("float32")
        # appending the image into the list
        train_img.append(img)

    labels_img = []
    for nb_image in range(1, nb_picture + 1):
        # defining the image path
        if nb_image < 10:
            image_path = "EM_ISBI_Challenge/EM_ISBI_Challenge/train_labels/labels_0{}.png".format(
                nb_image
            )
        if nb_image >= 10:
            image_path = (
                "EM_ISBI_Challenge/EM_ISBI_Challenge/train_labels/labels_{}.png".format(
                    nb_image
                )
            )
        # reading the image
        img = imread(image_path, as_gray=True)
        # normalizing the pixel values
        img = img / 255.0
        # converting the type of pixel to float 32
        img = img.astype("float32")
        # appending the image into the list
        labels_img.append(img)

    # converting the lists to numpy array
    train_x = np.array(train_img)
    label_x = np.array(labels_img)

    # converting training images into torch format
    train_x = train_x.reshape(nb_picture, 1, 512, 512)
    train_x = torch.from_numpy(train_x)

    label_x = label_x.reshape(nb_picture, 512, 512)
    label_x = torch.from_numpy(label_x)

    return (train_x, label_x)
