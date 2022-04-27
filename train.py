

from get_datas import get_datas
from network import UNet
from PIL import Image
# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ConvTranspose2d
from torch.optim import Adam, SGD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


train_x, label_x = get_datas(10)
#train_x1 = train_x_all[0:15]
#label_x1 = label_x_all[0:15]
m = nn.Softmax(dim=1)
loss = nn.CrossEntropyLoss()

network = UNet(10)

for p in range(0,10):
    print('Round {}'.format(p))
    #indices = torch.randperm(len(train_x1))[:10]
    #train_x = train_x1[indices]
    #label_x = label_x1[indices]

    result = network(train_x)
    result_soft =m(result)

    output = loss(result_soft, label_x.long())
    output.backward()

    image_result2 = result_soft[1]
    image_result2.shape
    one_label_result = torch.empty((512,512), dtype=torch.int64)
    for i in range(0,512):
        for j in range(0,512):
            if image_result2[0,i,j]>image_result2[1,i,j]:
                one_label_result[i,j] = 255
            if image_result2[0,i,j]<=image_result2[1,i,j]:
                one_label_result[i,j] = 0
    one_label_result =np.array(one_label_result).astype(np.uint8)
    im = Image.fromarray(one_label_result)
    im.save("results/result_{}.png".format(p))
torch.save(network, "results")
