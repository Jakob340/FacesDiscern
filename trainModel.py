# coding:utf-8
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import optimizers
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.models import load_model
from keras.utils import plot_model
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入图片的函数

def read_img(path):
    nameList = os.listdir(path)
    n = len(nameList)
    # indexImg,columnImg = plt.imread(path+'/'+nameList[0]).shape
    x_train = np.zeros([n,100,100,1]);y_train=[]
    for i in range(n):
        x_train[i,:,:,0] = imresize(plt.imread(path+'/'+nameList[i]),[100,100])
        y_train.append(np.int(nameList[i].split('.')[1]))
    return x_train,y_train

x_train,y_train = read_img('./dataset')
y_ture = y_train.copy()
y_train = pd.get_dummies(np.array(y_train))
print(y_train.shape)
n = len(y_train[y_train.iloc[:,1]==1])

x_train = np.array(x_train)

x_wzp = np.random.choice(y_train[y_train.iloc[:,0]==1].index.tolist(),n,replace=False)

x_train_w = x_train[x_wzp,:].copy()
x_train_l = x_train[y_train[y_train.iloc[:,1]==1].index.tolist()].copy()
x_train = np.concatenate([x_train_w,x_train_l],axis=0)

print(x_train.shape)

y_train = y_train.iloc[-208:,:].copy()
print(y_train.shape)
# 对两组数据进行洗牌
index = random.sample(range(len(y_train)),len(y_train))
index = np.array(index)
y_train = y_train.iloc[index,:]
# y_train.plot()
# plt.show()
x_train = x_train[index,:,:,:]





if __name__ == '__main__':
    model = Sequential()
    # 第一层：
    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    # model.add(Conv2D(64,(3,3),activation='relu'))
    # 第二层：
    # model.add(Conv2D(32, (3, 3), activation='relu'))  # model.add(Dropout(0.25))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # 2、全连接层和输出层：
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy' ,#'categorical_crossentropy',  # ,
                  optimizer=optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=1e-06),  # ,'Adadelta'
                  metrics=['accuracy'])
    # 模型训练
    model.fit(x_train, y_train, batch_size=104, epochs=50)
    # 模型得分
    score = model.evaluate(x_train, y_train, verbose=0)


    # 识别结果
    y_pred = model.predict(x_train)
    # 转onehot变label
    y_predict = np.argmax(y_pred,axis=1)

    # 精确度
    y_train = np.argmax(y_train.values,axis=1)
    accuracy = accuracy_score(y_train,y_predict)
    #打印score与accuracy
    print('score:',score,'  accuracy:',accuracy)
    # 检测结果
    plt.scatter(list(range(len(y_predict))),y_predict)

    model_dir = r'./trainer/imgmodel.h5'
    model_img = r'./trainer/imgKerasModel_ST.png'
    # 保存模型
    # model.save(model_dir)
    # 加载模型
    # model_keras = load_model(model_dir)
    # 画图
    # plot_model(model_keras,to_file=model_img)
    plt.show()
