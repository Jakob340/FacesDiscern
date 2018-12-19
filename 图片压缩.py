# coding:utf-8
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
import os
from scipy.misc import imresize
import numpy as np
from sklearn.utils import shuffle

# 压缩图片
def im_reduce(imgData,pixel=4):
    w,h = imgData.shape
    rgb = 1
    img = 0
    if rgb==3 or rgb == 1:
        img = np.array(imgData,dtype=float)/255
    else:
        img = imgData
    # img = imgData
    img1 = img.reshape(-1,rgb)
    x_train = shuffle(img1)

    # 聚类模型建立
    kmeans = KMeans(n_clusters=pixel)
    kmeans.fit(x_train[:1000,:])
    # 分别对原始图片进行分类
    x_class = kmeans.predict(img1)
    # 类别的中心点
    cluster_center = kmeans.cluster_centers_
    print(cluster_center)
    image = np.zeros([w,h,rgb])
    n = 0
    for i in range(w):
        for j in range(h):
            index = x_class[n]
            rgbColor = cluster_center[index,:]
            image[i,j,:] = rgbColor
            n+=1
    return image.reshape((w,h))


path = os.listdir('./dataset/')

img_path = os.path.join('./dataset',path[100])

data = plt.imread(img_path)


img_data_1 = imresize(data,[28,28])
plt.figure()
plt.imshow(data,cmap='gray')
img = np.array(im_reduce(data,pixel=2))
print(img.shape)
img_data_2 = imresize(img,[28,28])
# plt.figure(figsize=(2,2))
plt.imshow(img_data_2,cmap='gray')
# plt.xticks([])
# plt.yticks([])
plt.show()
# print(data.shape)
