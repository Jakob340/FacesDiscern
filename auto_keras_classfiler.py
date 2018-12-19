# coding:utf-8
import time
import matplotlib.pyplot as plt
from autokeras import ImageClassifier
from autokeras.utils import pickle_to_file,pickle_from_file
from keras.engine.saving import load_model
from keras.utils import plot_model
from scipy.misc import imresize
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入图片的函数

def read_img(path):
    nameList = os.listdir(path)
    n = len(nameList)
    # indexImg,columnImg = plt.imread(path+'/'+nameList[0]).shape
    x_train = np.zeros([n,28,28,1]);y_train=[]
    for i in range(n):
        x_train[i,:,:,0] = imresize(plt.imread(path+'/'+nameList[i]),[28,28])
        y_train.append(np.int(nameList[i].split('.')[1]))
    return x_train,y_train

x_train,y_train = read_img('./dataset')
y_train = pd.DataFrame(y_train)
n = len(y_train[y_train.iloc[:,0]==2])

x_train = np.array(x_train)

x_wzp = np.random.choice(y_train[y_train.iloc[:,0]==1].index.tolist(),n,replace=False)

x_train_w = x_train[x_wzp,:].copy()
x_train_l = x_train[y_train[y_train.iloc[:,0]==2].index.tolist()].copy()
x_train = np.concatenate([x_train_w,x_train_l],axis=0)

print(x_train.shape)

y_train = y_train.iloc[-208:,:].copy()

# 对两组数据进行洗牌
index = random.sample(range(len(y_train)),len(y_train))
index = np.array(index)
y_train = y_train.iloc[index,:]
# y_train.plot()
# plt.show()
x_train = x_train[index,:,:,:]



# x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2)
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# y_test = y_test.values.reshape(-1)
y_train = y_train.values.reshape(-1)

# 数据测试

'''
print(y_train)
for i in range(5):
    n = i*20
    img = x_train[n,:,:,:].reshape((28,28))
    print(y_train[n])
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
'''




if __name__=='__main__':
    start = time.time()
    # 模型构建
    model = ImageClassifier(verbose=True)
    # 搜索网络模型
    model.fit(x_train,y_train,time_limit=1*60)
    # 验证最优模型
    model.final_fit(x_train,y_train,x_train,y_train,retrain=True)
    # 给出评估结果
    score = model.evaluate(x_train,y_train)
    # 识别结果
    y_predict = model.predict(x_train)
    # y_pred = np.argmax(y_predict,axis=1)
    # 精确度
    accuracy = accuracy_score(y_train,y_predict)
    # 打印出score与accuracy
    print('score:',score,'  accuracy:',accuracy)
    print(y_predict,y_train)
    model_dir = r'./trainer/new_auto_learn_Model.h5'
    model_img = r'./trainer/imgModel_ST.png'

    # 保存可视化模型
    # model.load_searcher().load_best_model().produce_keras_model().save(model_dir)
    pickle_to_file(model,model_dir)
    # 加载模型
    # automodel = load_model(model_dir)
    # models = pickle_from_file(model_dir)
    # 输出模型 structure 图
    # plot_model(automodel, to_file=model_img)

    end = time.time()
    print('time:',end-start)
