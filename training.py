#coding:utf-8
import cv2

# 导入numpy进行矩阵计算
import numpy as np

# 导入Python图像库(PIL)
from PIL import Image

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# 创建用于人脸识别的局部二进制图形直方图
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 采用预建的人脸训练模型，进行人脸检测
detector = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")

# 创建获取图像和标签数据的方法
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # 创建一个空列表来存放人脸数据
    faceSamples=[]
    
    # 创建一个列表来存放人脸名称
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # 获取图像并将其转换为灰度
        PIL_img = cv2.cvtColor(cv2.imread(imagePath),cv2.COLOR_RGB2GRAY)#Image.open(imagePath).convert('L')

        # 把数据转换成numpy数组
        img_numpy = np.array(PIL_img,'uint8')

        # 获取图像的名称
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # 从训练图像中得到人脸
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # 将图像添加到面部样本中
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # 将ID添加到IDs中
            ids.append(id)

    # 传递face数组和IDs数组
    return faceSamples,ids

# 获取人脸和id
faces,ids = getImagesAndLabels('dataset')
# print(len(faces))
print(np.array(faces[3]).shape)
print(ids)
# 使用面部和id训练模型
recognizer.train(faces, np.array(ids))
x,y = recognizer.predict(faces[300])
print(x,100-y)
# 将模型保存到trainer.yml
# assure_path_exists('trainer/')
# recognizer.save('trainer/trainer.yml')
