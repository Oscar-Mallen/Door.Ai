import os
import pandas as pd
import numpy as np
import cv2 as cv

id_names = pd.read_csv('C:\\Users\\oscar\\OneDrive\\Desktop\\Door AI new\\id-names.csv')
id_names = id_names[['id', 'name']]

recognizer = cv.face.LBPHFaceRecognizer_create(threshold=500)


def create_train():
    faces = []
    labels = []
    for id in os.listdir('C:\\Users\\oscar\\OneDrive\\Desktop\\Door AI new\\faces'):
        path = os.path.join('C:\\Users\\oscar\\OneDrive\\Desktop\\Door AI new\\faces', id)
        try:
            os.listdir(path)
        except:
            continue
        for img in os.listdir(path):
            try:
                face = cv.imread(os.path.join(path, img))
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

                faces.append(face)
                labels.append(int(id))
            except:
                pass
    return np.array(faces), np.array(labels)


faces, labels = create_train()

print('Training Started')
recognizer.train(faces, labels)
recognizer.save('C:\\Users\\oscar\\OneDrive\\Desktop\\Door AI new\\Face-Recognition-with-OpenCV\\Classifiers\\trainedmode.xml')
print('Training Complete!')
