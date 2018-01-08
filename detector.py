# -- coding: utf8 -
import cv2
import sqlite3
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

def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM people WHERE id = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


# 显示在边框旁边的文字属性
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        profile = getProfile(id)
        if (profile != None):
            # 显示检测到的人脸对应的人名
            cv2.cv.PutText(cv2.cv.fromarray(img), str(round(conf,2)), (x, y + h), font, 255)
            cv2.cv.PutText(cv2.cv.fromarray(img), 'name:' + str(profile[1]), (x, y + h + 30), font, 255)
            cv2.cv.PutText(cv2.cv.fromarray(img), 'age:' + str(profile[2]), (x, y + h + 60), font, 255)
        else:
            cv2.cv.PutText(cv2.cv.fromarray(img), str('unkonw'), (x, y + h + 60), font, 255)

    cv2.imshow("Face",img);
    # 按Q键退出识别程序
    if(cv2.waitKey(2) == ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
