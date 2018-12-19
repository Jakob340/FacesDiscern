# coding:utf-8
import cv2
from keras.models import load_model
from autokeras.utils import pickle_from_file
# Import numpy for matrices calculations
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# 创建用于人脸识别的局部二进制图形直方图模型，也就是LBP算法模型
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# 载入训练好的模型
recognizer.read('trainer/trainer.yml')

# model = load_model('./trainer/imgmodel.h5')
model = pickle_from_file(r'./trainer/new_auto_learn_Model.h5')
# 导入模型分类器
cascadePath = "classifiers/haarcascade_frontalface_default.xml"

# 从预构建的模型中创建分类器
faceCascade = cv2.CascadeClassifier(cascadePath)

# 设定字体样式
font = cv2.FONT_HERSHEY_SIMPLEX

# 初始化并启动视频帧捕获
cam = cv2.VideoCapture(0)

# Loop
while True:
    # 读取视频帧
    ret, im =cam.read()

    # 将捕获的帧转换为灰度
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # 从视频框架得到所有的脸
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # 每一张脸
    for(x,y,w,h) in faces:

        # 在面部周围创建矩形
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # 识别一张属于哪个ID的脸
        # Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # 自己训练的模型
        train = cv2.resize(np.array(gray[y:y + h, x:x + w]),(28,28))
        xtrain = np.zeros([1,28,28,1])
        xtrain[0,:,:,0] = np.array(train)
        Id = model.predict(xtrain)
        # Id = np.argmax(Id,axis=1)
        print(Id[0],type(Id[0]))
        # 检查ID是否存在
        idlist = ['wuzaipei','lxl']
        Idd = Id[0]

        if(Id == 0 or Id == 1):
            # Id = "Nazmi {0:.2f}%".format(round(100 - confidence, 2))

            # 自己的模型
            Idd = "Nazmi "+idlist[Id[0]-1]
        cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(im, str(Idd), (x, y - 40), font, 1, (255, 255, 255), 3)

        # 用文字描述图画中的人物
        # cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        # cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # 用有界矩形显示视频帧
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
