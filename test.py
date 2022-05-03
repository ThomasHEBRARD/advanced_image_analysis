import pickle
import itertools
import statistics

import matplotlib.pyplot as plt

iou_score = []
cross_entropy_loss = []
train_L = []
valid_L = []
for epoch in range(34):
    train_file = open(f"lists2/train/train_logs_list_{epoch}.pkl", "rb")
    train_data = pickle.load(train_file)
    for f in train_data:
        f["epoch"] = epoch
    train_L += train_data

    valid_file = open(f"lists2/valid/valid_logs_list_{epoch}.pkl", "rb")
    valid_data = pickle.load(valid_file)
    for f in valid_data:
        f["epoch"] = epoch
    valid_L += valid_data

X_train, X_valid = [], []
Y_train, Y_valid = [], []
i = 0

grouped_train = itertools.groupby(train_L, lambda x: x["epoch"])
grouped_valid = itertools.groupby(valid_L, lambda x: x["epoch"])

for epoch, d in grouped_train:
    if epoch != 0:
        to_m = []
        for f in d:
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(f["iou_score"])
            else:
                print("prout")
        X_train.append(epoch)
        Y_train.append(statistics.mean(to_m))

for epoch, d in grouped_valid:
    if epoch != 0:
        to_m = []
        for f in d:
            print(f["iou_score"])
            if f["iou_score"] != 0.3243663012981415:
                to_m.append(f["iou_score"])
            else:
                print("prout")
        X_valid.append(epoch)
        Y_valid.append(statistics.mean(to_m))


plt.plot(X_train, Y_train, label = "train")
plt.plot(X_valid, Y_valid, label = "valid")
plt.title('iou_score')
plt.legend()
plt.show()
