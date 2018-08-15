####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')

# 每录入一张人脸的时候在这里写一个id，记住一点就是每个人的ID都不能相同。
face_id = 1

# Initialize sample face image
count = 0

assure_path_exists("dataset/")

# Start looping
while(True):

    # 捕获的视频帧 Capture video frame
    _, image_frame = vid_cam.read()

    # 帧转换为灰度图
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # 检测不同大小的帧，人脸矩形列表，返回四个值就是人脸位置的坐标
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # 将图像帧裁剪成矩形
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # 将捕获的图像保存到数据集文件夹中
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # 显示视频帧，在人的脸上有一个有边界的矩形
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # 如果拍摄的图像达到100，停止拍摄视频
    elif count>100:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
