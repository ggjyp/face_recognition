# -- coding: utf8 -
import os
import cv2
import numpy as np
from PIL import Image
# Describe: 训练识别器
# Author  : 江依鹏


# 创建LBP人脸识别器
recognizer = cv2.createLBPHFaceRecognizer();
path = 'dataSet'

# 遍历数据集
def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        faceNP = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(10)
    return np.array(IDs), faces


Ids, faces = getImageWithID(path)
# 训练
recognizer.train(faces,Ids)
# 保存训练结果
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
