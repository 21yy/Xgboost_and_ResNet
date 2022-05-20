import cv2
import xgboost as xgb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#

# image = cv2.imread("bird_10/train/AMERICAN AVOCET/AMERICAN AVOCET_train_101.jpg",0)
# print(image.shape)
# blue= cv2.GaussianBlur(image,(5,5),0)
# print(blue.shape)
#
# cv2.imshow("test",image)
# cv2.waitKey(0)
#
# image = cv2.pyrDown(image)
# cv2.imshow("test_2",image)
# cv2.waitKey(0)
#
# image = cv2.pyrDown(image)
# cv2.imshow("test_3",image)
# cv2.waitKey(0)
#
# image = cv2.pyrDown(image)
# cv2.imshow("test_4",image)
# cv2.waitKey(0)

# def calculatePCA(image):
#     # list_image = np.array([image.flatten()])
#     # print(list_image)
#     # print(list_image.shape)
#     # print(image.shape)
#     pca = PCA(n_components=1000)
#
#     result_image = pca.fit_transform(image)
#     # print(pca.explained_variance_ratio_)
#     return result_image


class_ind = {
    "AMERICAN AVOCET" : 0,
    "BALD EAGLE" : 1,
    "BLUE GROUSE" : 2,
    "COCKATOO" : 3,
    "DARK EYED JUNCO" : 4,
    "FLAME TANAGER" : 5,
    "GREEN JAY" : 6,
    "STRAWBERRY FINCH" : 7,
    "VERMILION FLYCATHER" : 8,
    "YELLOW HEADED BLACKBIRD" : 9
}




# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# rr = calculatePCA(X)
# print("test",rr.shape)
# pca = PCA(n_components=2)
# newX = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
# invX = pca.inverse_transform(X)
# print(X.shape)
# print(newX.shape)
def std_PCA(**kwargs):
    scalar = MinMaxScaler()  # 用于数据预处理(归一化和缩放)
    pca = PCA(**kwargs)  # PCA本身不包含预处理
    pipline = Pipeline([('scalar', scalar), ('pca', pca)])
    return pipline


def generateDatset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for each_class in os.listdir("bird_10/train"):
        for each_image in os.listdir("bird_10/train/" + each_class):
            path = os.path.join(os.getcwd(),"bird_10","train",each_class,each_image)
            image = cv2.imread(path,0)
            image= cv2.GaussianBlur(image,(5,5),0)
            # print(blue.shape)

            # cv2.imshow("test",image)
            # cv2.waitKey(0)

            image = cv2.pyrDown(image)
            image = cv2.pyrDown(image)
            # cv2.imshow("test_2",image)
            image = list(image.flatten())
            X_train.append(image)
            y_train.append(class_ind[each_class])

    # X_train = calculatePCA(np.array(X_train))
    mm_pca = std_PCA(n_components=100,svd_solver='randomized',whiten=False)
    mm_pca.fit(X_train)
    X_train = mm_pca.transform(X_train)


    for each_class in os.listdir("bird_10/test"):
        for each_image in os.listdir("bird_10/test/" + each_class):
            path = os.path.join(os.getcwd(),"bird_10","test",each_class,each_image)
            # image = np.array(cv2.imread(path,0)).flatten()
            # result_image = calculatePCA(image).flatten()
            image = cv2.imread(path, 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            # print(blue.shape)

            # cv2.imshow("test",image)
            # cv2.waitKey(0)

            image = cv2.pyrDown(image)
            image = cv2.pyrDown(image)
            # cv2.imshow("test_2",image)
            image = list(image.flatten())
            X_test.append(image)
            # X_test = calculatePCA(X_test)
            y_test.append(class_ind[each_class])
    # X_test = calculatePCA(X_test)
    X_test= mm_pca.transform(X_test)

    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)
#


X_train,y_train, X_test,y_test = generateDatset()
print(X_train.shape)

# d_train = xgb.DMatrix(X_train,label=y_train)
# d_test = xgb.DMatrix(X_test, label=y_test)
# param = {
#         'max_depth': 3,
#         'eta': 0.05,
#         'silent': 1,
#         'objective': 'multi:softmax',
#         'num_class': 10}
#
model = XGBClassifier(
    learning_rate=0.001,
    n_estimators=10000,
    max_depth=100,
    min_child_weight=0,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    scale_pos_weight=1,
    random_state=20
)
print("training start ")

model.fit(X_train, y_train, eval_set = [(X_test,y_test)],
          eval_metric = "mlogloss", early_stopping_rounds = 10,
          verbose = True)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test,y_pred)
print("准确率: %.2f%%" % (accuracy*100.0))
#
