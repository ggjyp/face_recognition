# -- coding: utf8 -
import cv2
# Describe: 训练识别器
# Author  : 江依鹏


# 导入openCV自带的人脸检测配置文件
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0)
# 创建LBP人脸识别器
rec = cv2.createLBPHFaceRecognizer();
# 加载训练结果至人脸识别器
rec.load('recognizer\\trainningData.yml')
id = 0
# 显示在边框旁边的文字属性
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,4)

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        # 设置人脸ID与人名的关联
        if(id==1):
            id='Rick'
        if(id==2):
            id='Yeah'
        if(id==3):
            id='Yellow'
        # 显示检测到的人脸对应的人名
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow("Face",img);
    # 按Q键退出识别程序
    if(cv2.waitKey(1) == ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
