import cv2
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans

DATA_PATH = "bird_10/"
WIDTH = 224
HEIGHT = 224


def select_categories(path):
    all_categories = os.listdir(path)
    cs = []
    for c in all_categories:
        amount = len(os.listdir(os.path.join(path, c)))
        cs.append((c, amount))
    cs.sort(key=lambda s: s[1], reverse=True)
    return [cat[0] for cat in cs]


# def get_feature(path, name='new_bird'):
#     # path = 'data/AFRICAN FIREFINCH/1.jpg'
#     maxCorners = 100
#     qualityLevel = 0.01
#     minDistance = 10
#
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('new_bird_g.jpg', np.reshape(gray, (224, 224, 1)))
#     corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
#
#     corners = np.int0(corners)
#     for i in corners:
#         x, y = i.ravel()
#         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
#
#     cv2.imwrite(name, img)
#     cv2.imshow('img', img)


def load_img(path):
    X = []
    y = []
    categories = select_categories(path)
    for i in range(len(categories)):
        c = categories[i]
        p = os.path.join(path, c)
        for filename in os.listdir(p):
            img = cv2.imread(os.path.join(p, filename))
            # X.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
            hist = cv2.calcHist([img], [0, 1, 2], None, [8] * 3, [0, 256] * 3)
            X.append(hist.ravel())
            y.append(i)
    return X, y


# def load_img2(path):
#     X = []
#     y = []
#     categories = select_categories(path)
#     for i in range(len(categories)):
#         c = categories[i]
#         p = os.path.join(path, c)
#         for filename in os.listdir(p):
#             img_o = cv2.imread(os.path.join(p, filename))
#             image_gray = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
#             img = cv2.Canny(image_gray, threshold1=20, threshold2=200)  # 224 *224
#             # X.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
#             # hist = cv2.calcHist([img], [0, 1, 2], None, [8] * 3, [0, 256] * 3)
#             X.append(img.ravel())
#             y.append(i)
#     return X, y


def load_img3(path):
    X = []
    y = []
    all_des = []
    all_imgs = []
    categories = select_categories(path)
    sift = cv2.xfeatures2d.SIFT_create(sigma=3, nOctaveLayers=10)
    for i in range(len(categories)):
        c = categories[i]
        p = os.path.join(path, c)

        for filename in os.listdir(p):
            full_path = os.path.join(p, filename)
            img = cv2.imread(full_path)
            kp, des = sift.detectAndCompute(img, None)
            all_imgs.append((full_path, i, img, kp, des))
            all_des.extend(des)

    # Clustering
    k = np.size(len(categories)) * 10
    batch_size = np.size(os.listdir(path)) * 3
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(all_des)

    # build histograms
    for full_path, label, img, kp, des in all_imgs:
        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp
        X.append(histo)
        y.append(label)
    return X, y


def load_data():
    # pca_transformer = PCA(n_components=110)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    path1 = os.path.join(DATA_PATH, 'train/')
    path2 = os.path.join(DATA_PATH, 'test/')
    path3 = os.path.join(DATA_PATH, 'val/')
    X_train_in, y_train_in = load_img3(path1)
    X_test_in, y_test_in = load_img3(path2)
    X_val_in, y_val_in = load_img3(path3)
    # X_train_val_in, X_test_in, y_train_val_in, y_test_in = train_test_split(np.array(X), np.array(y), test_size=0.1)
    # X_train_in, X_val_in, y_train_in, y_val_in = train_test_split(X_train_val_in, y_train_val_in, test_size=0.1)
    # X_train_in = pca_transformer.fit_transform(X_train_in)
    # X_val_in = pca_transformer.transform(X_val_in)
    # X_test_in = pca_transformer.transform(X_test_in)
    X_train.extend(X_train_in)
    y_train.extend(y_train_in)
    X_val.extend(X_val_in)
    y_val.extend(y_val_in)
    X_test.extend(X_test_in)
    y_test.extend(y_test_in)
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)


def train():
    cat_size = 30
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(X_train.shape, y_train.shape)
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_val, label=y_val)
    d_test = xgb.DMatrix(X_test, label=y_test)

    # TODO: need to come up with the paras
    param = {
        'max_depth': 2,
        'eta': 0.1,
        'subsample': 0.5,
        'silent': 1,
        'objective': 'multi:softmax',
        'num_class': cat_size}
    num_round = 20  # the number of training iterations
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    bst = xgb.train(param, d_train, 150, evallist, early_stopping_rounds=10)
    bst.save_model('xgb.model')
    y_test = d_test.get_label()
    y_pred = bst.predict(d_test)
    # y_pred = np.asarray([np.argmax(line) for line in bst.predict(d_test)])
    auc = accuracy_score(y_test, y_pred)
    print(auc)


train()
