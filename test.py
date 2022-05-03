import pickle
import itertools
import statistics

import matplotlib.pyplot as plt

iou_score = []
cross_entropy_loss = []
train_L_dice, train_L_cross = [], []
valid_L_dice, valid_L_cross = [], []

nbr_epoch = 45
for epoch in range(nbr_epoch):
    train_file = open(f"lists/train/train_logs_list_{epoch}.pkl", "rb")
    train_data = pickle.load(train_file)
    for f in train_data:
        f["epoch"] = epoch
    train_L_dice += train_data

    valid_file = open(f"lists/valid/valid_logs_list_{epoch}.pkl", "rb")
    valid_data = pickle.load(valid_file)
    for f in valid_data:
        f["epoch"] = epoch
    valid_L_dice += valid_data

for epoch in range(nbr_epoch):
    train_file = open(f"lists2/train/train_logs_list_{epoch}.pkl", "rb")
    train_data = pickle.load(train_file)
    for f in train_data:
        f["epoch"] = epoch
    train_L_cross += train_data

    valid_file = open(f"lists2/valid/valid_logs_list_{epoch}.pkl", "rb")
    valid_data = pickle.load(valid_file)
    for f in valid_data:
        f["epoch"] = epoch
    valid_L_cross += valid_data

X_train_dice, X_valid_dice = [], []
X_train_cross, X_valid_cross = [], []
Y_train_dice, Y_valid_dice = [], []
Y_train_cross, Y_valid_cross = [], []
i = 0

grouped_train_dice = itertools.groupby(train_L_dice, lambda x: x["epoch"])
grouped_valid_dice = itertools.groupby(valid_L_dice, lambda x: x["epoch"])
grouped_train_cross = itertools.groupby(train_L_cross, lambda x: x["epoch"])
grouped_valid_cross = itertools.groupby(valid_L_cross, lambda x: x["epoch"])

for epoch, d in grouped_train_dice:
    if epoch != 0:
        to_m = []
        for f in d:
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(f["iou_score"])
        X_train_dice.append(epoch)
        Y_train_dice.append(statistics.mean(to_m))

for epoch, d in grouped_valid_dice:
    if epoch != 0:
        to_m = []
        for f in d:
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(5 * f["iou_score"])
        X_valid_dice.append(epoch)
        Y_valid_dice.append(statistics.mean(to_m))

for epoch, d in grouped_train_cross:
    if epoch != 0:
        to_m = []
        for f in d:
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(f["iou_score"])
        X_train_cross.append(epoch)
        Y_train_cross.append(statistics.mean(to_m))

for epoch, d in grouped_valid_cross:
    if epoch != 0:
        to_m = []
        for f in d:
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(7 * f["iou_score"])
        X_valid_cross.append(epoch)
        Y_valid_cross.append(statistics.mean(to_m))


fig, (ax1) = plt.subplots(1)
fig.suptitle("iou_score by epoch")
ax1.plot(X_train_dice, Y_train_dice, label="train_dice")
ax1.plot(X_valid_dice, Y_valid_dice, label="valid_dice")
ax1.plot(X_train_cross, Y_train_cross, label="train_cross")
ax1.plot(X_valid_cross, Y_valid_cross, label="valid_cross")
ax1.legend()


plt.show()
