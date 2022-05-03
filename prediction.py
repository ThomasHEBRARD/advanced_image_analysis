import os
import torch
from get_datas_and_visu import (
    CreateDataset,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
    colour_code_segmentation,
    reverse_one_hot,
    visualize,
)
import numpy as np
import cv2

# We set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists("models/cross_model.pth"):
    best_model = torch.load("models/cross_model.pth", map_location=DEVICE)
    print("Loaded UNet model from this run.")

class_rgb_values = [[0, 0, 0], [255, 255, 255]]
select_class_rgb_values = np.array(class_rgb_values)

x_test_dir = "datas/test_images"
y_train_dir = "datas/test_labels"

#We build on dataset with the training augmentation and one without
test_dataset = CreateDataset(
    x_test_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

test_dataset_vis = CreateDataset(
    x_test_dir,
    y_train_dir,
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)


#We make sure that we have the right size of the image
def crop_image(image, target_image_dims=[1500, 1500, 3]):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding : image_size - padding,
        padding : image_size - padding,
        :,
    ]


for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = crop_image(test_dataset_vis[idx][0].astype("uint8"))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # We use our model for the prediction
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask, (1, 2, 0))

    # We create our hitmap
    pred_building_heatmap = pred_mask[:, :, 1]
    pred_mask = crop_image(
        colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    )

    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    gt_mask = crop_image(
        colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
    )
    
    #Then we load the visualization
    visualize(idx,
        ground_truth_mask=gt_mask,
        predicted_mask=pred_mask,
        predicted_building_heatmap=pred_building_heatmap,
    )
