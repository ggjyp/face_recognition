# -- coding: utf8 -
import cv2
# Describe: 采集人脸样本脚本
# Author  : 江依鹏


# 导入openCV自带的人脸检测配置文件
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# 启动电脑的摄像头
cam = cv2.VideoCapture(0)
# 待采集的人脸对应的编号
id = raw_input('Enter user id: ')
# 人脸样本编号
sampleNum = 0;
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1;
        # 将检测到的人脸保存至dataSet/User.{id}.{sampleNum}.jpg
        # 当id为1时保存后的文件名如:User.1.1.jpg
        cv2.imwrite('dataSet/User.'+str(id)+'.'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # 每隔0.1秒采集一次人脸图像
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    # 当采集了50个人脸样本后停止采集
    if (sampleNum >= 50):
        break
# 关闭摄像头
cam.release()
# 关闭摄像头显示窗口
cv2.destroyAllWindows()
