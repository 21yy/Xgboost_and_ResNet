import cv2
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score
import timeit

DATA_PATH = ''


def select_categories(path):
    all_categories = os.listdir(path)
    cs = []
    for c in all_categories:
        amount = len(os.listdir(os.path.join(path, c)))
        cs.append((c, amount))
    cs.sort(key=lambda s: s[1], reverse=True)
    return [cat[0] for cat in cs]


# load images in each category and extract features
def load_img(path):
    X = []
    y = []
    categories = select_categories(path)
    for i in range(len(categories)):
        c = categories[i]
        p = os.path.join(path, c)
        for filename in os.listdir(p):
            img = cv2.imread(os.path.join(p, filename))
            hist = cv2.calcHist([img], [0, 1, 2], None, [8] * 3, [0, 256] * 3)
            X.append(hist.ravel())
            y.append(i)
    return X, y


# load all data
def load_data():
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    path1 = os.path.join(DATA_PATH, 'train/')
    path2 = os.path.join(DATA_PATH, 'test/')
    path3 = os.path.join(DATA_PATH, 'val/')
    X_train_in, y_train_in = load_img(path1)
    X_test_in, y_test_in = load_img(path2)
    X_val_in, y_val_in = load_img(path3)
    X_train.extend(X_train_in)
    y_train.extend(y_train_in)
    X_val.extend(X_val_in)
    y_val.extend(y_val_in)
    X_test.extend(X_test_in)
    y_test.extend(y_test_in)
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)


# load data , train model and predict test set
def train():
    if len(DATA_PATH) == 0:
        print('data path is empty')
        return

    cat_size = len(os.listdir(os.path.join(DATA_PATH, 'train/')))
    print('\nTraining ', str(cat_size), ' categories...\n')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(X_train.shape, y_train.shape)
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    param = {
        'max_depth': 2,
        'eta': 0.1,
        'subsample': 0.5,
        'silent': 1,
        'objective': 'multi:softmax',
        'num_class': cat_size}
    num_round = 10
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    start = timeit.default_timer()

    bst = xgb.train(param, d_train, 150, evallist, early_stopping_rounds=num_round)
    stop = timeit.default_timer()
    total_time = stop - start
    bst.save_model('xgb_color' + str(cat_size) + '.json')
    y_test = d_test.get_label()
    y_pred = bst.predict(d_test)
    auc = accuracy_score(y_test, y_pred)
    print('\nResult of training ', str(cat_size), ' categories:\n')
    print("accuracy: %.2f%%" % (auc * 100.0))
    print("time: ", total_time)


# entry point
if __name__ == "__main__":
    dirs = ['bird_10', 'bird_15', 'bird_20', 'bird_30', 'bird_40', 'bird_50']
    for d in dirs:
        DATA_PATH = d
        train()
