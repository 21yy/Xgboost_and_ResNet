import torch
import torch.nn as nn
import torchvision.transforms as transforms
import xgboost as xgb
from model import resnet50
from PIL import Image
import numpy as np
import os
from sklearn.metrics import accuracy_score
import datetime
import sys

device = 'cpu'
net = resnet50()
model_weight_path = "resnet50_pretrained.pth"
massing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
net.fc = nn.Sequential()

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATA_PATH = sys.argv[1]


# extract feature
def generateDatset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    i = 0
    for each_class in os.listdir(os.path.join(DATA_PATH, "train")):
        print(each_class)
        for each_image in os.listdir(os.path.join(DATA_PATH, "train", each_class)):
            path = os.path.join(os.getcwd(), DATA_PATH, "train", each_class, each_image)
            img = data_transform(Image.open(path))
            img = torch.unsqueeze(img, dim=0)
            output = torch.squeeze(net(img)).detach().numpy().tolist()

            X_train.append(output)
            y_train.append(i)
        i = i + 1
    j = 0
    for each_class in os.listdir(os.path.join(DATA_PATH, "test")):
        print(each_class)
        for each_image in os.listdir(os.path.join(DATA_PATH, "test", each_class)):
            path = os.path.join(os.getcwd(), DATA_PATH, "test", each_class, each_image)
            img = data_transform(Image.open(path))
            img = torch.unsqueeze(img, dim=0)
            output = torch.squeeze(net(img)).detach().numpy().tolist()

            X_test.append(output)
            y_test.append(j)
        j = j + 1
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


X_train, y_train, X_test, y_test = generateDatset()

print(X_train.shape)

# train model
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)
cat_size = len(os.listdir(os.path.join(DATA_PATH, 'train/')))
print('\nTraining ', str(cat_size), ' categories...\n')
starttime = datetime.datetime.now()
param = {
    'max_depth': 2,
    'eta': 0.1,
    'silent': 1,
    'subsample': 0.5,
    'objective': 'multi:softmax',
    'colsample_bytree': 0.7,
    'lambda': 1,
    'num_class': cat_size}
num_round = 10
evallist = [(d_test, 'eval'), (d_train, 'train')]
bst = xgb.train(param, d_train, 150, evallist, early_stopping_rounds=num_round)
endtime = datetime.datetime.now()
total_time = (endtime - starttime).seconds
bst.save_model('xgb_res_' + str(cat_size) + '.json')

# predict
y_test = d_test.get_label()
y_pred = bst.predict(d_test)
auc = accuracy_score(y_test, y_pred)
print('\nResult of training ', str(cat_size), ' categories:\n')
print("accuracy: %.2f%%" % (auc * 100.0))
print("time: ", total_time)
