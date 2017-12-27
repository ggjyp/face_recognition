import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer();
path = 'dataSet'


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
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
